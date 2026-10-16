#!/usr/bin/env python3
"""Build the GSE200044 ND6 cell interface from the authors' official clusters."""

from pathlib import Path
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ASTER_DATA_ROOT", ROOT / "data")).resolve()
COHORT = os.environ.get("ASTER_COHORT", "ND6")
if COHORT not in {"ND6", "T2D6"}:
    raise ValueError("ASTER_COHORT must be ND6 or T2D6")
MULTIOME = DATA / COHORT / "multiome"
OFFICIAL = DATA / "reference/GSE200044/cell_cluster/multiome_RNA.h5ad"
OUT = DATA / COHORT / "stage2_multiome"
MIN_CELLS_PER_DONOR_CONTEXT = 10
MIN_VALID_DONORS = 3
MIN_CONTEXT_CELLS = 100

# Official cluster identities reproduced from Extended Data Fig. 7 and its
# reported cell counts. Non-endocrine labels follow the paper's marker scheme.
OFFICIAL_CELL_TYPES = {
    0: "Alpha",
    1: "Beta",
    2: "Acinar",
    3: "Delta",
    4: "Gamma",
    5: "Ductal",
    6: "Stellate",
    7: "Endothelial",
}


def decode(values):
    return np.asarray([
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ])


def read_official_cells() -> pd.DataFrame:
    with h5py.File(OFFICIAL, "r") as handle:
        obs = handle["obs"]
        cell_key = decode(obs["_index"][:])
        donor_categories = decode(obs["__categories/donor"][:])
        donors = donor_categories[np.asarray(obs["donor"][:], dtype=np.int64)]
        clusters = np.asarray(obs["cluster"][:], dtype=np.int64)
        frame = pd.DataFrame({
            "official_cell_key": cell_key,
            "sample_id": donors,
            "official_cluster": clusters,
            "official_rna_umi": np.asarray(obs["n_counts"][:]),
            "official_n_genes": np.asarray(obs["n_genes"][:]),
            "official_percent_mito": np.asarray(obs["percent_mito"][:]),
        })
    if set(frame["official_cluster"].unique()) != set(OFFICIAL_CELL_TYPES):
        raise ValueError("Official cluster set differs from the published eight clusters")
    frame["cell_type"] = frame["official_cluster"].map(OFFICIAL_CELL_TYPES)
    return frame


def cellranger_barcodes(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return decode(handle["matrix/barcodes"][:])


def build_metadata(samples: pd.DataFrame, official: pd.DataFrame):
    frames = []
    manifest = []
    for sample in samples.itertuples(index=False):
        outs = MULTIOME / "04_cellranger_arc" / sample.donor_id / "outs"
        h5 = outs / "filtered_feature_bc_matrix.h5"
        fragments = outs / "atac_fragments.tsv.gz"
        metrics_path = outs / "per_barcode_metrics.csv"
        required = [h5, fragments, Path(str(fragments) + ".tbi"), metrics_path]
        if not all(path.exists() for path in required):
            raise FileNotFoundError(f"Incomplete Cell Ranger output for {sample.donor_id}")

        donor_cells = official.loc[official["sample_id"].eq(sample.sample_id)].copy()
        prefix = f"{sample.sample_id}_"
        if not donor_cells["official_cell_key"].str.startswith(prefix).all():
            raise ValueError(f"Unexpected official barcode prefix for {sample.sample_id}")
        donor_cells["barcode"] = donor_cells["official_cell_key"].str[len(prefix):]

        cr_barcodes = cellranger_barcodes(h5)
        donor_cells = donor_cells.loc[donor_cells["barcode"].isin(cr_barcodes)].copy()
        if len(donor_cells) / max(1, official["sample_id"].eq(sample.sample_id).sum()) < 0.90:
            raise ValueError(f"Low official-to-GRCh38 barcode mapping for {sample.donor_id}")

        metric_columns = [
            "barcode", "is_cell", "gex_umis_count", "gex_genes_count",
            "atac_fragments", "atac_TSS_fragments",
            "atac_peak_region_fragments", "atac_mitochondrial_reads",
        ]
        metrics = pd.read_csv(metrics_path, usecols=metric_columns)
        metrics = metrics.loc[metrics["barcode"].isin(donor_cells["barcode"])]
        donor_cells = donor_cells.merge(
            metrics, on="barcode", how="left", validate="one_to_one"
        )
        if donor_cells[metric_columns[1:]].isna().any().any():
            raise ValueError(f"Missing Cell Ranger metrics for {sample.donor_id}")

        donor_cells.insert(0, "cell_id", sample.donor_id + "#" + donor_cells["barcode"])
        donor_cells["donor_id"] = sample.donor_id
        donor_cells["condition"] = sample.condition
        donor_cells["fragment_barcode"] = donor_cells["barcode"]
        donor_cells["qc_status"] = "pass_official"
        donor_cells["annotation_source"] = "GSE200044_official_multiome_cluster"
        frames.append(donor_cells)
        manifest.append({
            **sample._asdict(),
            "filtered_h5": str(h5),
            "fragments": str(fragments),
            "fragment_index": str(fragments) + ".tbi",
            "per_barcode_metrics": str(metrics_path),
            "n_official_cells": int(official["sample_id"].eq(sample.sample_id).sum()),
            "n_mapped_cells": len(donor_cells),
        })
    metadata = pd.concat(frames, ignore_index=True)
    if metadata["cell_id"].duplicated().any():
        raise ValueError("Duplicated cell_id")
    return metadata, pd.DataFrame(manifest)


def build_contexts(metadata: pd.DataFrame):
    donor_counts = (
        metadata.groupby(["official_cluster", "cell_type", "donor_id"], observed=True)
        .size().rename("n_cells").reset_index()
    )
    donor_counts["donor_context_valid"] = (
        donor_counts["n_cells"] >= MIN_CELLS_PER_DONOR_CONTEXT
    )
    valid_pairs = donor_counts.loc[donor_counts["donor_context_valid"]]

    contexts = (
        donor_counts.groupby(["official_cluster", "cell_type"], observed=True)
        .agg(n_cells=("n_cells", "sum"), n_donors=("donor_id", "nunique"))
        .reset_index()
    )
    valid_summary = (
        valid_pairs.groupby(["official_cluster", "cell_type"], observed=True)
        .agg(
            n_valid_cells=("n_cells", "sum"),
            n_valid_donors=("donor_id", "nunique"),
            min_cells_per_valid_donor=("n_cells", "min"),
            max_cells_per_valid_donor=("n_cells", "max"),
        ).reset_index()
    )
    contexts = contexts.merge(
        valid_summary, on=["official_cluster", "cell_type"], how="left"
    ).fillna(0)
    contexts["model_include"] = (
        (contexts["n_valid_cells"] >= MIN_CONTEXT_CELLS)
        & (contexts["n_valid_donors"] >= MIN_VALID_DONORS)
    )
    contexts = contexts.sort_values("official_cluster").reset_index(drop=True)
    contexts["cell_type_id"] = np.arange(len(contexts), dtype=np.int32)

    pair_keys = set(zip(valid_pairs["official_cluster"], valid_pairs["donor_id"]))
    metadata["donor_context_valid"] = [
        (cluster, donor) in pair_keys
        for cluster, donor in zip(metadata["official_cluster"], metadata["donor_id"])
    ]
    include = contexts.set_index("official_cluster")["model_include"]
    ids = contexts.set_index("official_cluster")["cell_type_id"]
    metadata["model_include"] = (
        metadata["donor_context_valid"]
        & metadata["official_cluster"].map(include).astype(bool)
    )
    metadata["cell_type_id"] = metadata["official_cluster"].map(ids).astype(np.int32)
    cell_map = metadata.loc[metadata["model_include"], [
        "cell_id", "barcode", "fragment_barcode", "donor_id", "sample_id",
        "condition", "cell_type_id", "cell_type", "official_cluster",
        "gex_umis_count", "gex_genes_count", "atac_fragments",
        "atac_TSS_fragments", "atac_peak_region_fragments",
    ]].copy()
    return metadata, contexts, donor_counts, cell_map


def plot_counts(donor_counts: pd.DataFrame, path: Path) -> None:
    table = donor_counts.pivot(
        index="cell_type", columns="donor_id", values="n_cells"
    ).fillna(0)
    table = table.loc[[OFFICIAL_CELL_TYPES[i] for i in sorted(OFFICIAL_CELL_TYPES)]]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    table.plot.bar(stacked=True, ax=axis, width=0.82)
    axis.set_xlabel("")
    axis.set_ylabel("Officially annotated cells")
    axis.legend(title="donor", ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(MULTIOME / "00_metadata/samples.tsv", sep="\t")
    condition = {"ND6": "Non-diabetic", "T2D6": "T2D"}[COHORT]
    if len(samples) != 6 or not samples["condition"].eq(condition).all():
        raise ValueError(f"Stage 2 expects exactly six {condition} donors")
    official = read_official_cells()
    metadata, manifest = build_metadata(samples, official)
    metadata, contexts, donor_counts, cell_map = build_contexts(metadata)
    if not contexts["model_include"].all():
        excluded = contexts.loc[~contexts["model_include"], "cell_type"].tolist()
        raise ValueError(f"Official contexts lack stable donor support: {excluded}")

    metadata.sort_values(["cell_type_id", "donor_id", "barcode"]).to_parquet(
        OUT / "cell_metadata.parquet", index=False, compression="zstd"
    )
    cell_map.sort_values(["cell_type_id", "donor_id", "barcode"]).to_parquet(
        OUT / "context_cell_map.parquet", index=False, compression="zstd"
    )
    contexts.to_csv(OUT / "model_contexts.tsv", sep="\t", index=False)
    donor_counts.to_csv(OUT / "context_donor_counts.tsv", sep="\t", index=False)
    manifest.to_csv(OUT / "sample_manifest.tsv", sep="\t", index=False)
    plot_counts(donor_counts, OUT / "stage2_cell_counts.png")
    print(
        f"stage2_pass official_cells={len(metadata)} model_cells={len(cell_map)} "
        f"contexts={len(contexts)}"
    )


if __name__ == "__main__":
    main()
