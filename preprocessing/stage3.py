#!/usr/bin/env python3
"""Stage 3: frozen Sei representations for the Stage 1 T2D variant index."""

import argparse
import gc
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import h5py
import numpy as np
import pandas as pd
import selene_sdk
import torch
from pyfaidx import Fasta
from selene_sdk.sequences import Genome
from selene_sdk.utils import load_path, parse_configs_and_run


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ASTER_DATA_ROOT", ROOT / "data")).resolve()
REF = DATA / "reference"
INDEX = DATA / "shared/stage1/model_variant_index.parquet"
FASTA = REF / "GRCh38_10x_ARC/fasta/genome.fa"
SEI = Path(os.environ.get("ASTER_SEI_DIR", ROOT / "software/sei-framework")).resolve()
OUT = DATA / "shared/stage3_sequence"
CHUNKS = OUT / ".chunks"
MODEL = SEI / "model/sei.pth"
PROJECTION = SEI / "model/projvec_targets.npy"
HISTONE = SEI / "model/histone_inds.npy"
BCFTOOLS = Path(os.environ.get("ASTER_BCFTOOLS", shutil.which("bcftools") or "bcftools"))
EXPECTED_COMMIT = "1e16bff9b1618af7065a23f82c4a3bb82aaeb689"
SAFE_COLUMNS = [
    "variant_row", "variant_key", "chr38", "pos38", "ref", "alt",
    "sequence_available", "atac_window_available",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SEI, text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Unexpected Sei commit: {commit}")
    for path in (INDEX, FASTA, Path(f"{FASTA}.fai"), MODEL, PROJECTION, HISTONE, BCFTOOLS):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)


def load_variants():
    variants = pd.read_parquet(INDEX).sort_values("variant_row").reset_index(drop=True)
    if variants.columns.tolist() != SAFE_COLUMNS:
        raise ValueError(f"Unexpected Stage 1 interface: {variants.columns.tolist()}")
    n = len(variants)
    if not np.array_equal(variants.variant_row.to_numpy(), np.arange(n)):
        raise ValueError("variant_row is not contiguous and zero-based")
    if variants.variant_key.duplicated().any() or not variants.sequence_available.all():
        raise ValueError("Stage 1 sequence index is not unique or contains unavailable rows")
    if not variants.chr38.astype(str).str.fullmatch(r"chr(?:[1-9]|1[0-9]|2[0-2])").all():
        raise ValueError("Only chr1-chr22 GRCh38 variants are accepted")
    if not variants.ref.str.fullmatch("[ACGT]").all() or not variants.alt.str.fullmatch("[ACGT]").all():
        raise ValueError("Stage 3 accepts biallelic SNVs only")
    if variants.ref.eq(variants.alt).any():
        raise ValueError("REF and ALT must differ")

    genome = Fasta(str(FASTA), as_raw=True, sequence_always_upper=True)
    n_ambiguous = np.empty(n, dtype="uint16")
    for i, row in enumerate(variants.itertuples(index=False)):
        p0 = int(row.pos38) - 1
        sequence = genome[row.chr38][p0 - 2048:p0 + 2048]
        if len(sequence) != 4096 or sequence[2048] != row.ref:
            raise ValueError(f"Reference mismatch at variant_row={row.variant_row}")
        n_ambiguous[i] = sum(base not in "ACGT" for base in sequence)
    return variants, n_ambiguous


def write_vcf(variants, path):
    lengths = {}
    with open(f"{FASTA}.fai") as handle:
        for line in handle:
            chrom, length, *_ = line.rstrip().split("\t")
            lengths[chrom] = int(length)
    chroms = sorted(variants.chr38.unique(), key=lambda x: int(x[3:]))
    with open(path, "w") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        for chrom in chroms:
            handle.write(f"##contig=<ID={chrom},length={lengths[chrom]}>\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for row in variants.itertuples(index=False):
            handle.write(
                f"{row.chr38}\t{row.pos38}\tvr_{row.variant_row}\t"
                f"{row.ref}\t{row.alt}\t.\tPASS\t.\n"
            )
    subprocess.run(
        [str(BCFTOOLS), "norm", "--check-ref", "e", "--fasta-ref", str(FASTA),
         "--output-type", "v", "--output", "/dev/null", str(path)],
        check=True,
    )


def replace_paths(value):
    if hasattr(value, "keywords"):
        replace_paths(value.keywords)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and "<PATH>" in item:
                value[key] = item.replace("<PATH>", str(SEI))
            else:
                replace_paths(item)


def run_sei(vcf, work, batch_size):
    profile_dir = work / "profiles"
    profile_dir.mkdir(parents=True)
    config = load_path(SEI / "model/sei_varianteffect_prediction.yml", instantiate=False)
    replace_paths(config)
    config["analyze_sequences"].keywords["batch_size"] = batch_size
    config["analyze_sequences"].keywords["write_mem_limit"] = 1000
    config["analyze_sequences"].bind(
        reference_sequence=Genome(str(FASTA)),
        use_cuda=True,
    )
    config["variant_effect_prediction"].update(
        vcf_files=[str(vcf)], output_dir=str(profile_dir), save_data=["predictions"]
    )
    parse_configs_and_run(config)
    prefix = vcf.stem
    return (
        profile_dir / f"{prefix}.ref_predictions.h5",
        profile_dir / f"{prefix}.alt_predictions.h5",
        profile_dir / f"{prefix}_row_labels.txt",
    )


def project_predictions(ref_file, alt_file, labels_file, variants):
    labels = pd.read_csv(labels_file, sep="\t")
    expected = [f"vr_{row}" for row in variants.variant_row]
    if labels["name"].tolist() != expected:
        raise ValueError("Sei output order differs from variant_row")
    if not labels["ref_match"].astype(str).str.lower().eq("true").all():
        raise ValueError("Sei reported a reference mismatch")
    if not labels["contains_unk"].astype(str).str.lower().eq("false").all():
        raise ValueError("Sei reported an ambiguous sequence")

    sys.path.insert(0, str(SEI))
    from utils import sc_hnorm_varianteffect, sc_projection

    vectors = np.load(PROJECTION)
    histone = np.load(HISTONE)
    with h5py.File(ref_file, "r") as ref_handle, h5py.File(alt_file, "r") as alt_handle:
        ref = ref_handle["data"][:]
        alt = alt_handle["data"][:]
    if ref.shape != (len(variants), 21907) or alt.shape != ref.shape:
        raise ValueError(f"Unexpected Sei prediction shape: {ref.shape}, {alt.shape}")
    if not np.isfinite(ref).all() or not np.isfinite(alt).all():
        raise ValueError("Non-finite Sei predictions")
    z_ref = sc_projection(ref, vectors)[:, :40].astype("float32")
    z_alt = sc_projection(alt, vectors)[:, :40].astype("float32")
    delta = sc_hnorm_varianteffect(ref, alt, vectors, histone).astype("float32")
    delta_swap = sc_hnorm_varianteffect(alt, ref, vectors, histone).astype("float32")
    if np.max(np.abs(delta + delta_swap)) >= 1e-6:
        raise ValueError("Allele-swap antisymmetry failed")
    return z_ref, z_alt, delta


def infer(variants, work, batch_size):
    work.mkdir(parents=True)
    vcf = work / "variants.vcf"
    write_vcf(variants, vcf)
    files = run_sei(vcf, work, batch_size)
    return project_predictions(*files, variants)


def smoke_variants(variants, n=16):
    selected = []
    for chrom in sorted(variants.chr38.unique(), key=lambda x: int(x[3:])):
        group = variants.loc[variants.chr38.eq(chrom)]
        selected.append(group.iloc[len(group) // 2])
        if len(selected) == n:
            return pd.DataFrame(selected)[SAFE_COLUMNS].sort_values("variant_row").reset_index(drop=True)
    raise RuntimeError("Unable to select smoke variants")


def run_smoke(variants):
    smoke = smoke_variants(variants)
    results = []
    for name, batch_size in (("b8", 8), ("b16a", 16), ("b16b", 16)):
        work = CHUNKS / f"smoke_{name}"
        results.append(infer(smoke, work, batch_size))
        shutil.rmtree(work)
        gc.collect()
        torch.cuda.empty_cache()
    for left, right in zip(results[1], results[2]):
        if np.max(np.abs(left - right)) >= 1e-6:
            raise ValueError("Repeated fixed-batch inference is not deterministic")
    pearson = np.corrcoef(results[0][2].ravel(), results[1][2].ravel())[0, 1]
    if not np.isfinite(pearson) or pearson <= 0.999999:
        raise ValueError(f"Cross-batch delta correlation failed: {pearson}")
    print(f"smoke_pass n={len(smoke)} chromosomes={smoke.chr38.nunique()} pearson={pearson:.9f}")


def chunk_path(chunk_id):
    return CHUNKS / f"chunk_{chunk_id:04d}.npz"


def valid_chunk(path, rows):
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            return (
                np.array_equal(data["variant_row"], rows) and
                data["z_ref"].shape == (len(rows), 40) and
                data["z_alt"].shape == (len(rows), 40) and
                data["delta"].shape == (len(rows), 40) and
                all(np.isfinite(data[key]).all() for key in ("z_ref", "z_alt", "delta"))
            )
    except Exception:
        return False


def run_chunks(variants, batch_size, chunk_size, worker_id=0, num_workers=1):
    for chunk_id, start in enumerate(range(0, len(variants), chunk_size)):
        if chunk_id % num_workers != worker_id:
            continue
        end = min(start + chunk_size, len(variants))
        chunk = variants.iloc[start:end].reset_index(drop=True)
        target = chunk_path(chunk_id)
        rows = chunk.variant_row.to_numpy("int64")
        if valid_chunk(target, rows):
            print(f"chunk_resume rows={start}:{end}", flush=True)
            continue
        work = CHUNKS / f"work_{chunk_id:04d}"
        shutil.rmtree(work, ignore_errors=True)
        z_ref, z_alt, delta = infer(chunk, work, batch_size)
        temporary = target.with_suffix(".npz.tmp")
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle, variant_row=rows, z_ref=z_ref, z_alt=z_alt, delta=delta
            )
        temporary.replace(target)
        shutil.rmtree(work)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"chunk_pass rows={start}:{end}", flush=True)


def combine(variants, infer_variants, n_ambiguous, batch_size, chunk_size, elapsed):
    n = len(variants)
    z_ref = np.zeros((n, 40), dtype="float32")
    z_alt = np.zeros((n, 40), dtype="float32")
    delta = np.zeros((n, 40), dtype="float32")
    for chunk_id, start in enumerate(range(0, len(infer_variants), chunk_size)):
        end = min(start + chunk_size, len(infer_variants))
        with np.load(chunk_path(chunk_id)) as data:
            expected_rows = infer_variants.iloc[start:end].variant_row.to_numpy("int64")
            if not np.array_equal(data["variant_row"], expected_rows):
                raise ValueError(f"Chunk row mismatch: {chunk_id}")
            z_ref[expected_rows] = data["z_ref"]
            z_alt[expected_rows] = data["z_alt"]
            delta[expected_rows] = data["delta"]
    x_seq = np.concatenate((z_ref, z_alt, delta, np.abs(delta)), axis=1)
    if x_seq.shape != (n, 160) or not np.isfinite(x_seq).all():
        raise ValueError("Invalid final sequence tensor")

    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / "sequence_sei.h5.tmp"
    result = OUT / "sequence_sei.h5"
    strings = h5py.string_dtype("utf-8")
    class_names = [line.strip() for line in open(SEI / "model/seqclass.names")][:40]
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("variant_row", data=variants.variant_row.to_numpy("int64"))
        handle.create_dataset("variant_key", data=variants.variant_key.to_numpy(), dtype=strings)
        handle.create_dataset("class_name", data=np.asarray(class_names, dtype=object), dtype=strings)
        for name, data in (("z_ref_raw", z_ref), ("z_alt_raw", z_alt), ("delta_hnorm", delta)):
            handle.create_dataset(name, data=data, chunks=(256, 40), compression="gzip", shuffle=True)
        handle.create_dataset("x_seq", data=x_seq, chunks=(256, 160), compression="gzip", shuffle=True)
        handle.create_dataset("n_ambiguous", data=n_ambiguous)
        available = n_ambiguous == 0
        reject_code = np.where(available, "", "ambiguous_reference_window")
        handle.create_dataset("available", data=available)
        handle.create_dataset("reject_code", data=reject_code.astype(object), dtype=strings)
        handle.attrs.update({
            "genome_build": "GRCh38",
            "sequence_length": 4096,
            "variant_offset_0based": 2048,
            "feature_definition": "[z_ref_raw,z_alt_raw,delta_hnorm,abs(delta_hnorm)]",
            "strand_ensemble": "forward_reverse_complement_mean",
            "sei_commit": EXPECTED_COMMIT,
            "reference_fasta_sha256": sha256(FASTA),
            "sei_model_sha256": sha256(MODEL),
            "projection_vectors_sha256": sha256(PROJECTION),
            "histone_indices_sha256": sha256(HISTONE),
            "pytorch_version": str(torch.__version__),
            "selene_sdk_version": str(getattr(selene_sdk, "__version__", "0.6.0")),
            "batch_size": batch_size,
            "inference_chunk_size": chunk_size,
            "elapsed_seconds": elapsed,
        })
    temporary.replace(result)
    with h5py.File(result, "r") as handle:
        if not np.array_equal(handle["variant_row"][:], np.arange(n)):
            raise ValueError("Final HDF5 row order mismatch")
        if handle["x_seq"].shape != (n, 160):
            raise ValueError("Final HDF5 tensor shape mismatch")
    shutil.rmtree(CHUNKS)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full", "combine"), default="full")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--no-combine", action="store_true")
    args = parser.parse_args()
    if args.chunk_size % args.batch_size:
        raise ValueError("chunk-size must be divisible by batch-size")
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")

    verify_inputs()
    variants, n_ambiguous = load_variants()
    infer_variants = variants.loc[n_ambiguous == 0].reset_index(drop=True)
    print(f"input_pass variants={len(variants)}")
    print(f"sequence_windows available={len(infer_variants)} rejected={(n_ambiguous > 0).sum()}")
    CHUNKS.mkdir(parents=True, exist_ok=True)
    if args.mode == "combine":
        result = combine(
            variants, infer_variants, n_ambiguous,
            args.batch_size, args.chunk_size, 0.0,
        )
        print(f"stage3_complete path={result} shape={len(variants)}x160")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Sei inference")
    torch.manual_seed(999)
    torch.cuda.manual_seed_all(999)
    np.random.seed(999)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    if not args.skip_smoke:
        run_smoke(infer_variants)
    if args.mode == "smoke":
        shutil.rmtree(CHUNKS)
        return
    started = time.time()
    run_chunks(
        infer_variants, args.batch_size, args.chunk_size,
        worker_id=args.worker_id, num_workers=args.num_workers,
    )
    if args.no_combine:
        print(f"worker_complete worker={args.worker_id}/{args.num_workers}")
        return
    if args.num_workers != 1:
        raise ValueError("Use --no-combine for sharded workers, then --mode combine")
    elapsed = round(time.time() - started, 3)
    result = combine(
        variants, infer_variants, n_ambiguous,
        args.batch_size, args.chunk_size, elapsed,
    )
    print(f"stage3_complete path={result} shape={len(variants)}x160 elapsed_seconds={elapsed}")


if __name__ == "__main__":
    main()
