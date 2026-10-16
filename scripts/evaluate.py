#!/usr/bin/env python3
"""Evaluate blind outer-fold predictions after joining held-out labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
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
    """Optional DIAMANTE statistical proxy; partly overlapping, never a training label."""
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


def published_metrics(root):
    locus = pd.read_csv(root / "oof_locus.tsv.gz", sep="\t")
    variants = pd.read_csv(root / "oof_variant_scores.tsv.gz", sep="\t")
    benchmark = pd.read_csv(root / "diamante_development_targets.tsv.gz", sep="\t")
    top = pd.read_csv(root / "top_context.tsv.gz", sep="\t")
    target_keys = benchmark.merge(variants[["mil_locus_id", "variant_row", "variant_key"]],
                                  on=["mil_locus_id", "variant_row"], validate="one_to_one")
    if len(target_keys) != len(benchmark) or not target_keys.variant_key_x.eq(
        target_keys.variant_key_y).all():
        raise ValueError("DIAMANTE target allele keys do not match the published OOF variants")
    labels = locus[["mil_locus_id", "y"]].drop_duplicates()
    index = (locus[["mil_locus_id", "match_set_id", "outer_fold"]].drop_duplicates()
             .rename(columns={"outer_fold": "fold"}))
    models = {"Sequence-only": "Sequence-only", "ASTER-ND6": "ASTER-ND6",
              "ASTER-T2D6": "ASTER-T2D6"}
    result = {"models": {}}
    for label, suffix in models.items():
        predictions = locus.loc[locus.model.eq(label),
                                ["mil_locus_id", "outer_fold", "logit", "probability"]]
        scores = variants[["mil_locus_id", "variant_row", f"score_{suffix}"]].rename(
            columns={f"score_{suffix}": "variant_score"})
        result["models"][label] = {
            **locus_metrics(predictions, labels, index),
            **variant_metrics(scores, benchmark),
        }
    paired = variants[["score_ASTER-ND6", "score_ASTER-T2D6"]]
    within = [spearmanr(group["score_ASTER-ND6"], group["score_ASTER-T2D6"]).statistic
              for _, group in variants.groupby("mil_locus_id") if len(group) >= 2]
    within = [value for value in within if pd.notna(value)]
    overlap = []
    for _, group in variants.groupby("mil_locus_id"):
        k = max(1, math.ceil(0.1 * len(group)))
        nd = set(group.nsmallest(k, "rank_ASTER-ND6").variant_row)
        t2d = set(group.nsmallest(k, "rank_ASTER-T2D6").variant_row)
        overlap.append(len(nd & t2d) / k)
    top1 = (variants.loc[variants["rank_ASTER-ND6"].eq(1), ["mil_locus_id", "variant_row"]]
            .merge(variants.loc[variants["rank_ASTER-T2D6"].eq(1),
                                ["mil_locus_id", "variant_row"]],
                   on=["mil_locus_id", "variant_row"]).mil_locus_id.nunique())
    endocrine = {"Alpha", "Beta", "Delta", "Gamma"}
    result["reference_concordance"] = {
        "n_bags_with_spearman": len(within),
        "global_spearman": float(paired.corr(method="spearman").iloc[0, 1]),
        "mean_within_bag_spearman": float(pd.Series(within).mean()),
        "mean_top10_percent_overlap": float(pd.Series(overlap).mean()),
        "top1_agreement": top1 / top.mil_locus_id.nunique(),
        "top_cell_agreement": float(variants.top_cell_type_nd6.eq(
            variants.top_cell_type_t2d6).mean()),
        "endocrine_top_context": {
            f"{model}_{label}": float(top.loc[top.y.eq(y), f"top_cell_type_{model}"].isin(endocrine).mean())
            for model in ("nd6", "t2d6") for label, y in (("positive", 1), ("control", 0))
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--published-results", type=Path,
                        help="Recompute manuscript metrics from the compact results/ tables")
    parser.add_argument("--variant-predictions", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.published_results:
        result = published_metrics(args.published_results)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if not args.data_root or not args.predictions:
        parser.error("--data-root and --predictions are required outside --published-results mode")
    root = args.data_root
    locus = pd.read_parquet(args.predictions)
    labels = pd.read_parquet(root / "stage7_dataset/labels.parquet")
    index = pd.read_parquet(root / "stage7_dataset/locus_index.parquet")
    result = {"locus": locus_metrics(locus, labels, index)}
    if args.benchmark:
        if not args.variant_predictions:
            parser.error("--benchmark requires --variant-predictions")
        benchmark = (pd.read_parquet(args.benchmark) if args.benchmark.suffix == ".parquet"
                     else pd.read_csv(args.benchmark, sep="\t"))
        if "eligible" in benchmark:
            benchmark = benchmark.loc[benchmark.eligible]
        if "benchmark_split" in benchmark:
            benchmark = benchmark.loc[benchmark.benchmark_split.eq("development")]
        if "variant_key" in benchmark:
            keys = pd.read_parquet(root / "stage1/model_variant_index.parquet",
                                   columns=["variant_row", "variant_key"])
            checked = benchmark.merge(keys, on="variant_row", validate="many_to_one",
                                      suffixes=("_benchmark", "_model"))
            if len(checked) != len(benchmark) or not checked.variant_key_benchmark.eq(
                checked.variant_key_model).all():
                raise ValueError("DIAMANTE target allele keys do not match Stage 1")
        result["variant_benchmark"] = variant_metrics(
            pd.read_parquet(args.variant_predictions), benchmark)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
