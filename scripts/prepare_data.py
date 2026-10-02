#!/usr/bin/env python3
"""Stage 7: convert Stage 1 bags plus Stage 6 folds into the PyTorch index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_index(root):
    source = root / "stage1"
    variants = pd.read_parquet(source / "model_variant_index.parquet").sort_values("variant_row")
    bags = pd.read_parquet(source / "locus_variant_map.parquet")
    labels = pd.read_parquet(source / "locus_labels.parquet")
    folds = pd.read_csv(source / "folds.tsv", sep="\t")
    if not np.array_equal(variants.variant_row.to_numpy(), np.arange(len(variants))):
        raise ValueError("Noncanonical variant_row")
    metadata = labels.merge(folds, on="mil_locus_id", validate="one_to_one")
    metadata = metadata.set_index("mil_locus_id")
    grouped = {key: value for key, value in bags.groupby("mil_locus_id")}
    starts, rows, offset = [], [], 0
    for locus in sorted(metadata.index.astype(str)):
        bag = grouped[locus].sort_values("bag_order", kind="stable")
        indices = bag.variant_row.to_numpy(np.int64)
        if not np.array_equal(bag.bag_order.to_numpy(), np.arange(len(bag))):
            raise ValueError(f"Noncontiguous bag order for {locus}")
        if len(set(indices)) != len(indices):
            raise ValueError(f"Repeated variant in {locus}")
        item = metadata.loc[locus]
        starts.append({"mil_locus_id": locus, "bag_start": offset, "bag_length": len(indices),
                       "fold": int(item.fold), "match_set_id": str(item.match_set_id),
                       "dependency_group": str(item.dependency_group),
                       "group_weight": float(item.group_weight), "trait_id": 0})
        rows.append(indices)
        offset += len(indices)
    index = pd.DataFrame(starts)
    labels = labels[["mil_locus_id", "y"]].sort_values("mil_locus_id").reset_index(drop=True)
    joined = index.merge(labels, on="mil_locus_id", validate="one_to_one")
    for group_id, group in joined.groupby("dependency_group"):
        positives = int(group.y.sum())
        if (len(group) != 3 * positives or group.fold.nunique() != 1
            or not np.allclose(group.group_weight, 1 / positives)):
            raise ValueError(f"Invalid dependency group: {group_id}")
        for _, match in group.groupby("match_set_id"):
            if len(match) != 3 or int(match.y.sum()) != 1 or match.bag_length.nunique() != 1:
                raise ValueError("Invalid 1:2 matched locus bags")
    return index, labels, np.concatenate(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    index, labels, bag_rows = build_index(root)
    alignment = json.loads((root / "stage6_alignment/alignment_manifest.json").read_text())
    contexts = pd.read_csv(root / "stage2_multiome/model_contexts.tsv", sep="\t")
    contexts = contexts.sort_values("cell_type_id")
    out = root / "stage7_dataset"
    out.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out / "locus_index.parquet", index=False)
    labels.to_parquet(out / "labels.parquet", index=False)
    np.save(out / "bag_variant_rows.npy", bag_rows.astype(np.int64), allow_pickle=False)
    (out / "dataset_manifest.json").write_text(json.dumps({
        "stage": 7, "model_input": "Sequence+ATAC Clean-G0",
        "n_loci": len(index), "n_positive_loci": int(labels.y.sum()),
        "n_control_loci": int((labels.y == 0).sum()),
        "n_match_sets": int(index.match_set_id.nunique()),
        "n_dependency_groups": int(index.dependency_group.nunique()),
        "n_unique_variants": alignment["n_variants"],
        "n_bag_memberships": len(bag_rows),
        "tensor_shapes": alignment["tensor_shapes"],
        "context_order": contexts.cell_type.astype(str).tolist(),
        "atac_feature_order": alignment["atac_feature_order"],
        "mask_semantics": alignment["mask_semantics"],
        "paths": {"sequence": "stage3_sequence/sequence_sei.h5",
                  "atac": "stage4_atac/atac_features.zarr",
                  "variant_index": "stage1/model_variant_index.parquet",
                  "cv_contexts": "stage6_alignment/cv_contexts.tsv",
                  "cv_scalers": "stage6_alignment/cv_scalers.npz"},
    }, indent=2) + "\n")
    print(f"Built {len(index)} bags from {len(bag_rows)} memberships")


if __name__ == "__main__":
    main()
