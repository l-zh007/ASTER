#!/usr/bin/env python3
"""Export compact, source-backed paper tables from the three OOF runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import locus_metrics, variant_metrics  # noqa: E402


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t")


def context_rows(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("outer_*/context/variant_contexts.parquet"))
    if len(paths) != 5:
        raise ValueError(f"Expected five context files under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Prepared ND6 root containing shared Stage 1 and Stage 7")
    for name in ("sequence-only", "nd6", "t2d6"):
        parser.add_argument(f"--{name}", type=Path, required=True,
                            help="Training output directory with both OOF parquet files")
    parser.add_argument("--benchmark", type=Path,
                        help="Allele-aligned DIAMANTE development benchmark, if available")
    parser.add_argument("--cell-interpretation", type=Path,
                        help="Existing paired variant-cell table; otherwise use OOF context inference files")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(args.data_root / "stage7_dataset/labels.parquet")
    index = pd.read_parquet(args.data_root / "stage7_dataset/locus_index.parquet")
    variants = pd.read_parquet(args.data_root / "stage1/model_variant_index.parquet",
                               columns=["variant_row", "variant_key"])
    roots = {"Sequence-only": args.sequence_only, "ASTER-ND6": args.nd6,
             "ASTER-T2D6": args.t2d6}
    locus_tables, score_tables, summary = [], [], []
    benchmark = read_table(args.benchmark) if args.benchmark else None
    if benchmark is not None:
        if "eligible" in benchmark:
            benchmark = benchmark.loc[benchmark.eligible]
        if "benchmark_split" in benchmark:
            benchmark = benchmark.loc[benchmark.benchmark_split.eq("development")]
        benchmark = benchmark[["mil_locus_id", "variant_row", "variant_key",
                               "statistical_target"]].drop_duplicates()
        checked = benchmark.merge(variants, on="variant_row", validate="many_to_one",
                                  suffixes=("_benchmark", "_model"))
        if len(checked) != len(benchmark) or not checked.variant_key_benchmark.eq(
            checked.variant_key_model).all():
            raise ValueError("DIAMANTE target allele keys do not match Stage 1")
        benchmark.to_csv(args.output / "diamante_development_targets.tsv.gz", sep="\t", index=False)
    for name, root in roots.items():
        locus = pd.read_parquet(root / "oof_locus_predictions.parquet")
        variant = pd.read_parquet(root / "oof_variant_predictions.parquet")
        metrics = locus_metrics(locus, labels, index)
        if benchmark is not None:
            metrics.update(variant_metrics(variant, benchmark))
        summary.append({"model": name, **{key: value for key, value in metrics.items()
                                          if key != "fold_metrics"}})
        locus = (locus[["mil_locus_id", "outer_fold", "logit", "probability"]]
                 .merge(labels, on="mil_locus_id", validate="one_to_one")
                 .merge(index[["mil_locus_id", "match_set_id", "dependency_group"]],
                        on="mil_locus_id", validate="one_to_one"))
        locus.insert(0, "model", name)
        locus_tables.append(locus)
        scores = variant[["mil_locus_id", "variant_row", "variant_score"]].copy()
        scores = scores.rename(columns={"variant_score": f"score_{name}"})
        scores[f"rank_{name}"] = (scores.groupby("mil_locus_id")[f"score_{name}"]
                                  .rank(ascending=False, method="min").astype(int))
        score_tables.append(scores)
    pd.concat(locus_tables, ignore_index=True).to_csv(
        args.output / "oof_locus.tsv.gz", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(args.output / "metrics.tsv", sep="\t", index=False)

    keys = ["mil_locus_id", "variant_row"]
    scores = score_tables[0]
    for table in score_tables[1:]:
        scores = scores.merge(table, on=keys, validate="one_to_one")
    if len(scores) != 106_938:
        raise ValueError("The three OOF runs do not share the paper's variant-bag memberships")
    scores = scores.merge(variants, on="variant_row", validate="many_to_one")
    scores = scores.sort_values(keys)
    if args.cell_interpretation:
        context_source = pd.read_parquet(args.cell_interpretation)
        for cohort, name in (("ND6", "nd6"), ("T2D6", "t2d6")):
            contexts = context_source.loc[context_source.cohort.eq(cohort),
                                          keys + ["top_cell_type"]]
            scores = scores.merge(contexts.rename(columns={"top_cell_type": f"top_cell_type_{name}"}),
                                  on=keys, validate="one_to_one")
    else:
        for name, root in (("nd6", args.nd6), ("t2d6", args.t2d6)):
            contexts = context_rows(root)[keys + ["top_cell_type"]]
            scores = scores.merge(contexts.rename(columns={"top_cell_type": f"top_cell_type_{name}"}),
                                  on=keys, validate="one_to_one")
    scores.to_csv(args.output / "oof_variant_scores.tsv.gz", sep="\t", index=False)

    size = scores.groupby("mil_locus_id").variant_row.transform("size")
    denominator = (size - 1).clip(lower=1)
    percentile_nd6 = 1 - (scores["rank_ASTER-ND6"] - 1) / denominator
    percentile_t2d6 = 1 - (scores["rank_ASTER-T2D6"] - 1) / denominator
    consensus = (percentile_nd6 + percentile_t2d6) / 2
    top = scores.assign(consensus=consensus).sort_values(
        ["mil_locus_id", "consensus", "variant_row"], ascending=[True, False, True]
    ).drop_duplicates("mil_locus_id")
    top = top[["mil_locus_id", "variant_row", "variant_key"]].merge(
        labels, on="mil_locus_id", validate="one_to_one")
    top = top.merge(scores[keys + ["top_cell_type_nd6", "top_cell_type_t2d6"]],
                    on=keys, validate="one_to_one")
    if len(top) != len(labels):
        raise ValueError("Consensus top variants lack cell-context assignments")
    top.to_csv(args.output / "top_context.tsv.gz", sep="\t", index=False)
    cases = []
    for region, rsid, locus, key in (
        ("GLIS3", "rs4237150", "MIL_POS_T2DGGI_SIGNAL_0628", "chr9:4290085:G:C"),
        ("PROX1", "rs17712208", "MIL_POS_T2DGGI_SIGNAL_0070", "chr1:213977102:T:A"),
    ):
        row = scores.loc[scores.mil_locus_id.eq(locus) & scores.variant_key.eq(key)]
        if len(row) != 1:
            raise ValueError(f"Missing paper case {rsid}")
        cases.append({"region": region, "rsid": rsid, "mil_locus_id": locus,
                      "variant_key_grch38": key,
                      "bag_size": int(size.loc[row.index[0]]),
                      "rank_nd6": int(row.iloc[0]["rank_ASTER-ND6"]),
                      "rank_t2d6": int(row.iloc[0]["rank_ASTER-T2D6"]),
                      "top_cell_nd6": row.iloc[0].top_cell_type_nd6,
                      "top_cell_t2d6": row.iloc[0].top_cell_type_t2d6})
    pd.DataFrame(cases).to_csv(args.output / "cases.tsv", sep="\t", index=False)
    print(json.dumps({"n_loci": len(labels), "n_memberships": len(scores),
                      "models": [row["model"] for row in summary]}, indent=2))


if __name__ == "__main__":
    main()
