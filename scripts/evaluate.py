#!/usr/bin/env python3
"""Evaluate blind outer-fold predictions after joining held-out labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def locus_metrics(predictions, labels, index):
    frame = predictions.merge(labels[["mil_locus_id", "y"]], on="mil_locus_id",
                              how="inner", validate="one_to_one")
    frame = frame.merge(index[["mil_locus_id", "match_set_id", "fold"]],
                        on="mil_locus_id", validate="one_to_one")
    if len(frame) != len(predictions) or (frame.fold != frame.outer_fold).any():
        raise ValueError("OOF predictions, labels and folds do not align")
    if frame.groupby("match_set_id").size().ne(3).any():
        raise ValueError("Incomplete matched sets in predictions")
    frame["rank"] = frame.groupby("match_set_id").logit.rank(ascending=False, method="average")
    positive_rank = frame.loc[frame.y.eq(1), "rank"]
    metrics = {
        "n_loci": len(frame),
        "auroc": roc_auc_score(frame.y, frame.probability),
        "auprc": average_precision_score(frame.y, frame.probability),
        "log_loss": log_loss(frame.y, frame.probability),
        "brier": brier_score_loss(frame.y, frame.probability),
        "matched_top1": (positive_rank == 1).mean(),
        "matched_mrr": (1 / positive_rank).mean(),
        "fold_metrics": {},
    }
    for fold, group in frame.groupby("outer_fold"):
        metrics["fold_metrics"][str(fold)] = {
            "n_loci": len(group), "auroc": roc_auc_score(group.y, group.probability),
            "auprc": average_precision_score(group.y, group.probability),
        }
    return metrics


def variant_metrics(predictions, benchmark):
    """Optional independent statistical-target benchmark, never a training label."""
    required = {"mil_locus_id", "variant_row", "statistical_target"}
    if not required <= set(benchmark):
        raise ValueError(f"Benchmark requires {sorted(required)}")
    scores = predictions[["mil_locus_id", "variant_row", "variant_score"]].copy()
    scores["rank"] = scores.groupby("mil_locus_id").variant_score.rank(
        ascending=False, method="average")
    bag_sizes = scores.groupby("mil_locus_id").size()
    merged = scores.merge(benchmark[["mil_locus_id", "variant_row", "statistical_target"]],
                          on=["mil_locus_id", "variant_row"],
                          how="inner", validate="one_to_one")
    records = []
    for locus, group in merged.groupby("mil_locus_id"):
        target = group.statistical_target.astype(bool)
        if bag_sizes[locus] < 2 or not target.any():
            continue
        best = float(group.loc[target, "rank"].min())
        records.append({"mil_locus_id": locus, "mrr": 1 / best,
                        "recall_at_1": float(best <= 1), "recall_at_3": float(best <= 3),
                        "recall_at_5": float(best <= 5)})
    result = pd.DataFrame(records)
    return {"n_benchmark_loci": len(result),
            **{key: float(result[key].mean()) for key in ("mrr", "recall_at_1",
                                                          "recall_at_3", "recall_at_5")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--variant-predictions", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    locus = pd.read_parquet(args.predictions)
    labels = pd.read_parquet(root / "stage7_dataset/labels.parquet")
    index = pd.read_parquet(root / "stage7_dataset/locus_index.parquet")
    result = {"locus": locus_metrics(locus, labels, index)}
    if args.benchmark:
        if not args.variant_predictions:
            parser.error("--benchmark requires --variant-predictions")
        benchmark = pd.read_parquet(args.benchmark)
        if "eligible" in benchmark:
            benchmark = benchmark.loc[benchmark.eligible]
        if "benchmark_split" in benchmark:
            benchmark = benchmark.loc[benchmark.benchmark_split.eq("development")]
        result["variant_benchmark"] = variant_metrics(
            pd.read_parquet(args.variant_predictions), benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
