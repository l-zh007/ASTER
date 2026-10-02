#!/usr/bin/env python3
"""Stage 6: align branch rows and fit train-fold-only Sequence/ATAC scalers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr


class Moments:
    def __init__(self, width):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, len(self.mean))
        if len(x):
            mean = x.mean(0)
            m2 = np.einsum("ij,ij->j", x - mean, x - mean)
            self.merge_values(len(x), mean, m2)

    def merge_values(self, count, mean, m2):
        if not count:
            return
        if not self.count:
            self.count, self.mean, self.m2 = count, mean.copy(), m2.copy()
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def merge(self, other):
        self.merge_values(other.count, other.mean, other.m2)

    def finish(self):
        if not self.count:
            raise ValueError("Cannot fit scaler without observations")
        scale = np.sqrt(np.maximum(self.m2 / self.count, 0))
        near_zero = scale < 1e-8
        scale[near_zero] = 1.0
        return self.mean, scale, np.full(len(scale), self.count), near_zero


def digest_lines(values):
    return hashlib.sha256("".join(f"{x}\n" for x in values).encode()).hexdigest()


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path)
    args = parser.parse_args()
    root = args.data_root
    sequence_file = args.sequence_file or root / "stage3_sequence/sequence_sei.h5"
    variants = pd.read_parquet(root / "stage1/model_variant_index.parquet").sort_values("variant_row")
    membership = pd.read_parquet(root / "stage1/locus_variant_map.parquet")
    folds = pd.read_csv(root / "stage1/folds.tsv", sep="\t")
    loci = pd.read_parquet(root / "stage1/locus_labels.parquet")
    n = len(variants)
    assert np.array_equal(variants.variant_row.to_numpy(), np.arange(n))
    assigned = membership[["mil_locus_id", "variant_row"]].merge(
        folds[["mil_locus_id", "fold"]], on="mil_locus_id", validate="many_to_one")
    variant_folds = assigned[["variant_row", "fold"]].drop_duplicates().sort_values("variant_row")
    if len(variant_folds) != n or variant_folds.variant_row.duplicated().any():
        raise ValueError("A variant occurs across dependency-safe folds")
    rows_by_fold = {fold: variant_folds.loc[variant_folds.fold.eq(fold), "variant_row"].to_numpy(np.int64)
                    for fold in range(5)}

    atac = zarr.open_group(str(root / "stage4_atac/atac_features.zarr"), mode="r")
    contexts = pd.read_csv(root / "stage2_multiome/model_contexts.tsv", sep="\t")
    contexts = contexts.sort_values("cell_type_id")
    if (not np.array_equal(contexts.cell_type_id.to_numpy(), np.arange(8))
        or not np.array_equal(atac["context_id"][:], np.arange(8))):
        raise ValueError("Official cell-context order is not the expected eight classes")
    with h5py.File(sequence_file, "r") as seq:
        keys = variants.variant_key.astype(str).to_numpy()
        if (seq["x_seq"].shape != (n, 160) or atac["x_atac"].shape != (n, 8, 100, 2)
            or not np.array_equal(decode(seq["variant_key"][:]), keys)
            or not np.array_equal(decode(atac["variant_key"][:]), keys)
            or not np.array_equal(seq["variant_row"][:], np.arange(n))
            or not np.array_equal(atac["variant_row"][:], np.arange(n))):
            raise ValueError("Sequence/ATAC variant_row or allele key alignment failed")
        seq_names = decode(seq["class_name"][:]).tolist()
        seq_stats, atac_stats = {}, {}
        for fold, rows in rows_by_fold.items():
            seq_stats[fold], atac_stats[fold] = Moments(160), Moments(2)
            for start in range(0, len(rows), 1024):
                part = rows[start:start + 1024]
                x_seq = np.asarray(seq["x_seq"][part])
                available = np.asarray(seq["available"][part], dtype=bool)
                seq_stats[fold].update(x_seq[available])
                x_atac = np.asarray(atac["x_atac"].oindex[part, :, :, :])
                mask = np.asarray(atac["atac_mask"].oindex[part, :], dtype=bool)
                atac_stats[fold].update(x_atac[mask].reshape(-1, 2))

    all_folds = set(range(5))
    contexts, scaler_keys = [], set()
    for outer in range(5):
        for inner in sorted(all_folds - {outer}):
            fit = tuple(sorted(all_folds - {outer, inner}))
            contexts.append((f"outer_{outer}__inner_{inner}", "inner_model_selection",
                             outer, inner, fit, (inner,), (outer,)))
            scaler_keys.add(fit)
        fit = tuple(sorted(all_folds - {outer}))
        contexts.append((f"outer_{outer}__refit", "outer_refit", outer, None,
                         fit, (), (outer,)))
        scaler_keys.add(fit)
    contexts.extend([
        ("fixed_split", "fixed_split_alias", 0, 1, (2, 3, 4), (1,), (0,)),
        ("final_all_loci", "final_fit_only", None, None, (0, 1, 2, 3, 4), (), ()),
    ])
    scaler_keys.update({(2, 3, 4), (0, 1, 2, 3, 4)})
    scaler_keys = sorted(scaler_keys, key=lambda x: (len(x), x))
    scaler_index = {key: i for i, key in enumerate(scaler_keys)}
    outputs = {f"{branch}_{name}": [] for branch in ("seq", "atac")
               for name in ("mean", "scale", "count", "near_zero")}
    for fit in scaler_keys:
        for branch, stats, width in (("seq", seq_stats, 160), ("atac", atac_stats, 2)):
            total = Moments(width)
            for fold in fit:
                total.merge(stats[fold])
            for name, value in zip(("mean", "scale", "count", "near_zero"), total.finish()):
                outputs[f"{branch}_{name}"].append(value)
    table = []
    locus_fold = folds.set_index("mil_locus_id").fold
    for name, role, outer, inner, fit, valid, test in contexts:
        fit_loci = sorted(locus_fold.index[locus_fold.isin(fit)].astype(str))
        fit_rows = np.unique(np.concatenate([rows_by_fold[fold] for fold in fit]))
        table.append({"cv_context_id": name, "role": role, "outer_fold": outer,
                      "inner_fold": inner, "fit_folds": ",".join(map(str, fit)),
                      "validation_folds": ",".join(map(str, valid)),
                      "test_folds": ",".join(map(str, test)),
                      "evaluation_allowed": int(role != "final_fit_only"),
                      "scaler_index": scaler_index[fit],
                      "fit_locus_sha256": digest_lines(fit_loci),
                      "fit_variant_sha256": digest_lines(keys[fit_rows])})
    out = root / "stage6_alignment"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(out / "cv_contexts.tsv", sep="\t", index=False)
    np.savez_compressed(out / "cv_scalers.npz",
                        scaler_fit_folds=np.asarray([",".join(map(str, key)) for key in scaler_keys]),
                        **{key: np.stack(value) for key, value in outputs.items()})
    (out / "alignment_manifest.json").write_text(json.dumps({
        "stage": 6, "n_variants": n, "n_loci": len(loci),
        "tensor_shapes": {"sequence": [n, 160], "atac": [n, 8, 100, 2]},
        "sequence_feature_names": seq_names,
        "atac_feature_order": ["donor_equal_log1p_cpm_accessibility",
                               "donor_equal_cell_support_fraction"],
        "mask_semantics": {"sequence_mask": "unavailable rows are zeroed after scaling",
                           "atac_mask": "observed variant-cell ATAC window"},
        "scaling": "per-feature train-fold-only population mean/std over unique variants",
    }, indent=2) + "\n")
    print(f"Built {len(contexts)} CV contexts and {len(scaler_keys)} train-only scalers")


if __name__ == "__main__":
    main()
