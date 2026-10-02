"""Portable Stage 7 reader for the final Sequence+ATAC experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset


def _folds(value) -> tuple[int, ...]:
    return () if pd.isna(value) else tuple(int(float(x)) for x in str(value).split(",") if x)


def _read_rows(array, rows: np.ndarray, *, zarr_array=False) -> np.ndarray:
    order = np.argsort(rows, kind="stable")
    values = np.asarray(array.oindex[rows[order]] if zarr_array else array[rows[order]])
    restored = np.empty_like(values)
    restored[order] = values
    return restored


class LocusDataset(Dataset):
    """One variable-length bag per item; test labels are never exposed."""

    def __init__(self, data_root: str | Path, context: str, partition: str,
                 sequence_file: str | Path | None = None):
        root = Path(data_root).expanduser().resolve()
        self.root, self.context, self.partition = root, context, partition
        self.sequence_path = (Path(sequence_file).expanduser().resolve() if sequence_file
                              else root / "stage3_sequence/sequence_sei.h5")
        self.atac_path = root / "stage4_atac/atac_features.zarr"
        table = pd.read_csv(root / "stage6_alignment/cv_contexts.tsv", sep="\t")
        rows = table.loc[table.cv_context_id.eq(context)]
        if len(rows) != 1:
            raise ValueError(f"Unknown CV context: {context}")
        self.cv = rows.iloc[0]
        roles = {"inner_model_selection": ("train", "validation"),
                 "outer_refit": ("train", "test"),
                 "fixed_split_alias": ("train", "validation", "test"),
                 "final_fit_only": ("train",)}
        if partition not in roles[str(self.cv.role)]:
            raise ValueError(f"Partition {partition} is not allowed for {context}")
        fold_column = {"train": "fit_folds", "validation": "validation_folds",
                       "test": "test_folds"}[partition]
        selected_folds = _folds(self.cv[fold_column])
        self.index = pd.read_parquet(root / "stage7_dataset/locus_index.parquet")
        self.index = self.index.loc[self.index.fold.isin(selected_folds)].reset_index(drop=True)
        if self.index.empty:
            raise ValueError(f"No bags in {context}/{partition}")
        self.include_labels = partition != "test"
        self.label_map = None
        if self.include_labels:
            labels = pd.read_parquet(root / "stage7_dataset/labels.parquet")
            self.label_map = labels.set_index("mil_locus_id").y
        self.bag_rows = np.load(root / "stage7_dataset/bag_variant_rows.npy", mmap_mode="r")
        variants = pd.read_parquet(root / "stage1/model_variant_index.parquet",
                                   columns=["variant_row", "variant_key"]).sort_values("variant_row")
        if not np.array_equal(variants.variant_row.to_numpy(), np.arange(len(variants))):
            raise ValueError("variant_row must be contiguous and zero-based")
        self.variant_keys = variants.variant_key.astype(str).to_numpy()
        with np.load(root / "stage6_alignment/cv_scalers.npz") as scalers:
            scaler = int(self.cv.scaler_index)
            if str(scalers["scaler_fit_folds"][scaler]) != str(self.cv.fit_folds):
                raise ValueError("Scaler was not fitted on the selected training folds")
            self.seq_mean = scalers["seq_mean"][scaler].astype(np.float32)
            self.seq_scale = scalers["seq_scale"][scaler].astype(np.float32)
            self.atac_mean = scalers["atac_mean"][scaler].astype(np.float32)
            self.atac_scale = scalers["atac_scale"][scaler].astype(np.float32)
        self._pid = self._seq = self._atac = None

    def __len__(self):
        return len(self.index)

    def _open(self):
        if self._pid == os.getpid():
            return
        self.close()
        self._seq = h5py.File(self.sequence_path, "r")
        self._atac = zarr.open_group(str(self.atac_path), mode="r")
        self._pid = os.getpid()

    def close(self):
        if getattr(self, "_seq", None) is not None:
            self._seq.close()
        self._seq = self._atac = self._pid = None

    def __del__(self):
        self.close()

    def __getitem__(self, index):
        self._open()
        record = self.index.iloc[index]
        start, length = int(record.bag_start), int(record.bag_length)
        rows = np.array(self.bag_rows[start:start + length], dtype=np.int64, copy=True)
        seq = _read_rows(self._seq["x_seq"], rows).astype(np.float32)
        seq_mask = _read_rows(self._seq["available"], rows).astype(bool)
        atac = _read_rows(self._atac["x_atac"], rows, zarr_array=True).astype(np.float32)
        cell_mask = _read_rows(self._atac["atac_mask"], rows, zarr_array=True).astype(bool)
        seq = (seq - self.seq_mean) / self.seq_scale
        atac = (atac - self.atac_mean) / self.atac_scale
        seq[~seq_mask] = 0.0
        atac[~cell_mask] = 0.0
        item = {
            "seq": torch.from_numpy(seq), "sequence_mask": torch.from_numpy(seq_mask),
            "atac": torch.from_numpy(atac), "cell_mask": torch.from_numpy(cell_mask),
            "variant_row": torch.from_numpy(rows),
            "group_weight": torch.tensor(float(record.group_weight), dtype=torch.float32),
            "trait_id": torch.tensor(int(record.trait_id)),
            "locus_id": str(record.mil_locus_id),
        }
        if self.include_labels:
            item["label"] = torch.tensor(float(self.label_map.loc[item["locus_id"]]))
        return item


class CachedDataset(Dataset):
    """Cache scaled items once per CV context to avoid repeated HDF5/Zarr reads."""

    def __init__(self, source: LocusDataset):
        self.index, self.label_map = source.index, source.label_map
        self.include_labels, self.partition = source.include_labels, source.partition
        self.items = [source[i] for i in range(len(source))]
        source.close()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items):
    b, m = len(items), max(x["seq"].shape[0] for x in items)
    out = {
        "seq": torch.zeros((b, m, 160)), "atac": torch.zeros((b, m, 8, 100, 2)),
        "sequence_mask": torch.zeros((b, m), dtype=torch.bool),
        "cell_mask": torch.zeros((b, m, 8), dtype=torch.bool),
        "variant_mask": torch.zeros((b, m), dtype=torch.bool),
        "variant_row": torch.full((b, m), -1, dtype=torch.long),
        "group_weight": torch.stack([x["group_weight"] for x in items]),
        "trait_id": torch.stack([x["trait_id"] for x in items]),
        "locus_id": [x["locus_id"] for x in items],
    }
    if all("label" in x for x in items):
        out["label"] = torch.stack([x["label"] for x in items])
    for i, item in enumerate(items):
        n = len(item["seq"])
        for key in ("seq", "atac", "sequence_mask", "cell_mask", "variant_row"):
            out[key][i, :n] = item[key]
        out["variant_mask"][i, :n] = True
    return out


@dataclass(frozen=True)
class Group:
    group_id: str
    match_sets: tuple[tuple[int, ...], ...]


def dependency_groups(dataset, *, shuffle=False, seed=2026, epoch=0):
    if not dataset.include_labels:
        raise ValueError("Dependency groups require train/validation labels")
    table = dataset.index.copy()
    table["y"] = table.mil_locus_id.map(dataset.label_map)
    groups = []
    for group_id, group in table.groupby("dependency_group", sort=True):
        matches = []
        for _, match in group.groupby("match_set_id", sort=True):
            match = match.sort_values(["y", "mil_locus_id"], ascending=[False, True])
            if len(match) != 3 or int(match.y.sum()) != 1:
                raise ValueError(f"Invalid 1:2 matched set in {group_id}")
            matches.append(tuple(int(i) for i in match.index))
        groups.append(Group(str(group_id), tuple(matches)))
    order = np.arange(len(groups))
    if shuffle:
        np.random.default_rng(seed + epoch).shuffle(order)
    for i in order:
        yield groups[i]


def match_batches(dataset, group):
    for indices in group.match_sets:
        yield collate([dataset[i] for i in indices])


def group_denominator(dataset, group, positive_weight):
    indices = [i for batch in group.match_sets for i in batch]
    records = dataset.index.iloc[indices]
    labels = torch.tensor(records.mil_locus_id.map(dataset.label_map).to_numpy(), dtype=torch.float32)
    weights = torch.tensor(records.group_weight.to_numpy(), dtype=torch.float32)
    return (weights * ((1 - labels) + positive_weight * labels)).sum()
