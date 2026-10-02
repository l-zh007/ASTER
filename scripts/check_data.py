#!/usr/bin/env python3
"""Confirm the frozen Stage 1/3/4/6/7 feature contract before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import LocusDataset, collate  # noqa: E402
from src.model import ASTER  # noqa: E402


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path)
    args = parser.parse_args()
    root = args.data_root
    manifest = json.loads((root / "stage7_dataset/dataset_manifest.json").read_text())
    assert manifest["atac_feature_order"] == [
        "donor_equal_log1p_cpm_accessibility", "donor_equal_cell_support_fraction"]
    variants = pd.read_parquet(root / "stage1/model_variant_index.parquet")
    keys = variants.sort_values("variant_row").variant_key.astype(str).to_numpy()
    with h5py.File(args.sequence_file or root / "stage3_sequence/sequence_sei.h5") as seq:
        if not np.array_equal(decode(seq["variant_key"][:]), keys):
            raise ValueError("Sequence and Stage 1 variant alleles do not align")
    atac = zarr.open_group(str(root / "stage4_atac/atac_features.zarr"), mode="r")
    if not np.array_equal(decode(atac["variant_key"][:]), keys):
        raise ValueError("ATAC and Stage 1 variant alleles do not align")
    contexts = pd.read_csv(root / "stage2_multiome/model_contexts.tsv", sep="\t")
    if (not np.array_equal(contexts.sort_values("cell_type_id").cell_type_id.to_numpy(), np.arange(8))
        or not np.array_equal(atac["context_id"][:], np.arange(8))):
        raise ValueError("The eight ATAC contexts do not follow the official order")
    dataset = LocusDataset(root, "outer_0__refit", "train", args.sequence_file)
    batch = collate([dataset[0]])
    assert batch["seq"].shape[-1] == 160 and batch["atac"].shape[-3:] == (8, 100, 2)
    assert not batch["seq"][~batch["sequence_mask"]].any()
    model = ASTER({"sequence_features": 160, "sequence_hidden": 128,
                   "atac_contexts": 8, "atac_bins": 100, "atac_channels": 2,
                   "atac_hidden": 64, "fusion_hidden": 64,
                   "tau_cell": 1.0, "tau_variant": 1.0}, 0.15).eval()
    with torch.inference_mode():
        output = model(**{key: batch[key] for key in model.input_keys})
    assert torch.isfinite(output["logit"]).all()
    print(f"Ready: {manifest['n_loci']} bags, {manifest['n_unique_variants']} variants; "
          f"model parameters: {sum(p.numel() for p in model.parameters())}")
    dataset.close()


if __name__ == "__main__":
    main()
