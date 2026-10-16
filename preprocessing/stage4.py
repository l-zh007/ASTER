#!/usr/bin/env python3
"""Stage 4: donor-balanced local ATAC profiles from GSE200044 ND6/T2D6."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import pandas as pd
import zarr


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ASTER_DATA_ROOT", ROOT / "data")).resolve()
COHORT = os.environ.get("ASTER_COHORT", "ND6")
if COHORT not in {"ND6", "T2D6"}:
    raise ValueError("ASTER_COHORT must be ND6 or T2D6")
CODE = Path(__file__).resolve().parent
STAGE1 = DATA / "shared/stage1"
STAGE2 = DATA / COHORT / "stage2_multiome"
OUT = DATA / COHORT / "stage4_atac"
BLACKLIST = Path(os.environ.get("ASTER_BLACKLIST", DATA / "reference/hg38.blacklist.bed.gz"))
N_BINS = 100


def load_inputs():
    variants = pd.read_parquet(STAGE1 / "model_variant_index.parquet").sort_values("variant_row")
    cells = pd.read_parquet(STAGE2 / "context_cell_map.parquet")
    contexts = pd.read_csv(STAGE2 / "model_contexts.tsv", sep="\t").sort_values("cell_type_id")
    if not np.array_equal(variants.variant_row.to_numpy(), np.arange(len(variants))):
        raise ValueError("variant_row must be zero-based and contiguous")
    if variants.variant_key.duplicated().any() or not variants.atac_window_available.all():
        raise ValueError("Stage 1 ATAC variant interface is invalid")
    if contexts.cell_type_id.tolist() != list(range(len(contexts))):
        raise ValueError("Stage 2 context order is not 0..7")
    manifest = pd.read_csv(STAGE2 / "sample_manifest.tsv", sep="\t")
    fragments = dict(zip(manifest.donor_id, manifest.fragments.map(Path)))
    if len(contexts) != 8:
        raise ValueError("The borrowed C++ counter expects the eight official GSE200044 clusters")
    if set(cells.donor_id) != set(fragments):
        raise ValueError("Unexpected donor set")
    if not all(path.exists() and Path(str(path) + ".tbi").exists() for path in fragments.values()):
        raise FileNotFoundError("A fragment file or tabix index is missing")
    expected = {"variant_row", "variant_key", "chr38", "pos38"}
    if not expected.issubset(variants.columns):
        raise ValueError("Missing Stage 1 variant columns")
    return variants.reset_index(drop=True), cells, contexts, fragments


def build_counter(binary: Path):
    subprocess.run(
        ["g++", "-O3", "-std=c++17", str(CODE / "stage4_count.cpp"), "-lz", "-o", str(binary)],
        check=True,
    )


def write_counter_inputs(variants, cells, work):
    variant_file = work / "variants.tsv"
    variant_table = pd.DataFrame({
        "chrom": variants.chr38.str.removeprefix("chr").astype(int),
        "pos0": variants.pos38.astype(np.int64) - 1,
        "variant_row": variants.variant_row,
    })
    variant_table.to_csv(variant_file, sep="\t", index=False, header=False)

    cell_files = {}
    for donor, frame in cells.groupby("donor_id", sort=True):
        frame = frame.sort_values(["cell_type_id", "fragment_barcode"]).copy()
        frame["local_index"] = np.arange(len(frame), dtype=np.int64)
        target = work / f"{donor}.cells.tsv"
        frame[["fragment_barcode", "cell_type_id", "local_index"]].to_csv(
            target, sep="\t", index=False, header=False
        )
        cell_files[donor] = target

    import gzip
    blacklist_file = work / "blacklist.bed"
    with gzip.open(BLACKLIST, "rt") as source, blacklist_file.open("w") as sink:
        shutil.copyfileobj(source, sink)
    return variant_file, cell_files, blacklist_file


def run_donor(binary, donor, fragment_file, variant_file, cell_file, blacklist_file, work):
    counts = work / f"{donor}.counts.bin"
    support = work / f"{donor}.support.bin"
    stats = work / f"{donor}.stats.tsv"
    subprocess.run(
        [str(binary), str(variant_file), str(cell_file), str(blacklist_file),
         str(fragment_file), str(counts), str(support), str(stats)],
        check=True,
    )
    return donor, counts, support, stats


def create_store(path, variants, contexts):
    n_contexts = len(contexts)
    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("x_atac", shape=(len(variants), n_contexts, N_BINS, 2),
                        chunks=(512, 1, N_BINS, 2), dtype="f4", compressor=compressor)
    root.create_dataset("atac_mask", data=np.ones((len(variants), n_contexts), dtype=bool),
                        chunks=(4096, 1), compressor=compressor)
    root.create_dataset("variant_row", data=variants.variant_row.to_numpy("int64"),
                        chunks=(8192,), compressor=compressor)
    root.create_dataset("variant_key", data=variants.variant_key.to_numpy(dtype=object),
                        chunks=(8192,), dtype=object, object_codec=numcodecs.VLenUTF8(), compressor=compressor)
    root.create_dataset("context_id", data=contexts.cell_type_id.to_numpy("int32"), compressor=compressor)
    root.attrs["channels"] = ["donor_equal_log1p_cpm_accessibility",
                              "donor_equal_cell_support_fraction"]
    return root


def aggregate(results, variants, contexts, output_store):
    n = len(variants)
    n_contexts = len(contexts)
    shape = (n, n_contexts, N_BINS)
    stats = []
    maps = {}
    for donor, counts_file, support_file, stats_file in results:
        maps[donor] = (
            np.memmap(counts_file, dtype="uint32", mode="r", shape=shape),
            np.memmap(support_file, dtype="uint32", mode="r", shape=shape),
        )
        frame = pd.read_csv(stats_file, sep="\t")
        frame.insert(0, "donor_id", donor)
        stats.append(frame)
    library = pd.concat(stats, ignore_index=True).merge(
        contexts[["cell_type_id", "cell_type"]], left_on="context_id", right_on="cell_type_id",
        validate="many_to_one",
    ).drop(columns="cell_type_id")
    qc_rows = []
    for context in range(n_contexts):
        accessibility = np.zeros((n, N_BINS), dtype=np.float64)
        cell_support = np.zeros((n, N_BINS), dtype=np.float64)
        valid_donors = library.loc[
            library.context_id.eq(context)
            & library.n_cells.gt(0)
            & library.total_valid_cuts.gt(0), "donor_id"
        ].tolist()
        if len(valid_donors) < 3:
            raise ValueError(f"Context {context} has fewer than three usable donors")
        for donor in valid_donors:
            row = library[(library.donor_id == donor) & (library.context_id == context)].iloc[0]
            counts, support = maps[donor]
            accessibility += np.log1p(1e6 * counts[:, context, :] / float(row.total_valid_cuts))
            cell_support += support[:, context, :] / float(row.n_cells)
        accessibility = (accessibility / len(valid_donors)).astype("float32")
        cell_support = (cell_support / len(valid_donors)).astype("float32")
        output_store["x_atac"][:, context, :, 0] = accessibility
        output_store["x_atac"][:, context, :, 1] = cell_support
        values = accessibility[accessibility > 0]
        supported = cell_support[cell_support > 0]
        qc_rows.append({
            "context_id": context,
            "cell_type": contexts.loc[contexts.cell_type_id == context, "cell_type"].iloc[0],
            "n_valid_donors": len(valid_donors),
            "nonzero_accessibility_fraction": float(np.mean(accessibility > 0)),
            "nonzero_support_fraction": float(np.mean(cell_support > 0)),
            "accessibility_median_nonzero": float(np.median(values)) if values.size else 0.0,
            "accessibility_p99": float(np.quantile(accessibility, 0.99)),
            "support_median_nonzero": float(np.median(supported)) if supported.size else 0.0,
            "support_p99": float(np.quantile(cell_support, 0.99)),
        })
    return library, pd.DataFrame(qc_rows)


def plot_qc(qc, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh(qc.cell_type, qc.nonzero_accessibility_fraction, color="#4477AA")
    axes[0].set_xlabel("Nonzero variant-bin fraction")
    axes[0].set_title("ATAC accessibility coverage")
    axes[1].barh(qc.cell_type, qc.support_p99, color="#228833")
    axes[1].set_xlabel("99th percentile")
    axes[1].set_title("Cell-support fraction")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    variants, cells, contexts, fragments = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    target_store = OUT / "atac_features.zarr"
    if target_store.exists():
        raise FileExistsError(f"Stage 4 output already exists: {target_store}")
    with tempfile.TemporaryDirectory(prefix="t2d_stage4_") as temporary:
        work = Path(temporary)
        binary = work / "stage4_count"
        build_counter(binary)
        variant_file, cell_files, blacklist_file = write_counter_inputs(variants, cells, work)
        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(
                    run_donor, binary, donor, fragments[donor], variant_file,
                    cell_files[donor], blacklist_file, work
                ): donor
                for donor in sorted(fragments)
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"counted {result[0]}", flush=True)

        temporary_store = OUT / "atac_features.zarr.tmp"
        target_store = OUT / "atac_features.zarr"
        if temporary_store.exists():
            shutil.rmtree(temporary_store)
        store = create_store(temporary_store, variants, contexts)
        library, qc = aggregate(results, variants, contexts, store)
        if not np.isfinite(store["x_atac"][:]).all():
            raise ValueError("ATAC tensor contains NaN or Inf")
        if store["x_atac"][:, :, :, 1].min() < 0 or store["x_atac"][:, :, :, 1].max() > 1:
            raise ValueError("Cell-support channel is outside [0,1]")
        temporary_store.replace(target_store)

    library[["donor_id", "context_id", "cell_type", "n_cells", "total_valid_cuts"]].to_csv(
        OUT / "library_sizes.tsv", sep="\t", index=False
    )
    qc.to_csv(OUT / "atac_qc.tsv", sep="\t", index=False)
    plot_qc(qc, OUT / "stage4_atac_qc.pdf")
    print(f"stage4_pass variants={len(variants)} contexts={len(contexts)}")


if __name__ == "__main__":
    main()
