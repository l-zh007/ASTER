#!/usr/bin/env python3
"""Recreate the four quantitative Fig. 2 panels from published source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/fig2.png"))
    args = parser.parse_args()
    loci = pd.read_csv(args.results / "oof_locus.tsv.gz", sep="\t")
    variants = pd.read_csv(args.results / "oof_variant_scores.tsv.gz", sep="\t")
    top = pd.read_csv(args.results / "top_context.tsv.gz", sep="\t")
    colors = {"Sequence-only": "#8A8F98", "ASTER-ND6": "#2166AC", "ASTER-T2D6": "#D18F32"}
    styles = {"Sequence-only": "-", "ASTER-ND6": "-", "ASTER-T2D6": "--"}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), constrained_layout=True)
    for model in ("Sequence-only", "ASTER-ND6", "ASTER-T2D6"):
        frame = loci.loc[loci.model.eq(model)]
        fpr, tpr, _ = roc_curve(frame.y, frame.probability)
        precision, recall, _ = precision_recall_curve(frame.y, frame.probability)
        axes[0, 0].plot(fpr, tpr, styles[model], color=colors[model],
                        lw=2 if model == "ASTER-ND6" else 1.6,
                        label=f"{model}  {roc_auc_score(frame.y, frame.probability):.3f}")
        axes[0, 1].plot(recall, precision, styles[model], color=colors[model],
                        lw=2 if model == "ASTER-ND6" else 1.6,
                        label=f"{model}  {average_precision_score(frame.y, frame.probability):.3f}")
    axes[0, 0].plot([0, 1], [0, 1], color="0.8", lw=1)
    axes[0, 0].set(xlabel="False positive rate", ylabel="True positive rate",
                   title="A  OOF bag discrimination")
    axes[0, 1].axhline(loci.y.mean(), color="0.8", lw=1)
    axes[0, 1].set(xlabel="Recall", ylabel="Precision", title="B  OOF precision–recall")
    for ax in axes[0]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8, loc="lower right" if ax is axes[0, 0] else "upper right")

    correlations = [spearmanr(group["score_ASTER-ND6"], group["score_ASTER-T2D6"]).statistic
                    for _, group in variants.groupby("mil_locus_id") if len(group) >= 2]
    correlations = pd.Series(correlations).dropna()
    axes[1, 0].hist(correlations, bins=30, color=colors["ASTER-ND6"], alpha=0.85)
    axes[1, 0].axvline(correlations.mean(), color="#C24B3B", lw=1.5)
    axes[1, 0].set(xlabel="Within-bag Spearman correlation", ylabel="Bags",
                   title=f"C  Reference ranking concordance (n={len(correlations):,})")
    axes[1, 0].text(0.03, 0.95, f"Mean = {correlations.mean():.3f}",
                    transform=axes[1, 0].transAxes, va="top")

    cell_types = ["Alpha", "Beta", "Delta", "Gamma", "Acinar", "Ductal", "Stellate", "Endothelial"]
    palette = ["#3976A4", "#2D9A91", "#75B7A4", "#A6C9A0",
               "#D5A051", "#BA7761", "#8C79AA", "#9FA5AC"]
    strata = [("ND6 positive", "nd6", 1), ("ND6 control", "nd6", 0),
              ("T2D6 positive", "t2d6", 1), ("T2D6 control", "t2d6", 0)]
    for row, (label, cohort, y) in enumerate(strata):
        cells = top.loc[top.y.eq(y), f"top_cell_type_{cohort}"]
        frequency = cells.value_counts(normalize=True).reindex(cell_types, fill_value=0)
        left = 0.0
        for cell, color in zip(cell_types, palette):
            value = float(frequency[cell])
            axes[1, 1].barh(row, value, left=left, color=color, height=0.68,
                            label=cell if row == 0 else None)
            left += value
        axes[1, 1].text(1.01, row, f"Endocrine {frequency.iloc[:4].sum():.1%}",
                        va="center", fontsize=8)
    axes[1, 1].set_yticks(range(4), [x[0] for x in strata])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0, 1.23)
    axes[1, 1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[1, 1].set_xlabel("Fraction of bags")
    axes[1, 1].set_title("D  Consensus-variant top context")
    axes[1, 1].legend(ncol=4, frameon=False, fontsize=7, loc="lower center",
                      bbox_to_anchor=(0.5, -0.34))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
