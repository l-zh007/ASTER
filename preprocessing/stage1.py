#!/usr/bin/env python3
"""T2DGGI 2024 Stage 1: signal-level MIL bags, matched controls and folds."""

from __future__ import annotations

import gzip
import hashlib
import math
import os
import pickle
import shutil
import subprocess
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import polars as pl
import pysam
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260901
THREADS = 12
CHROM_WORKERS = 4
POPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
ANCESTRIES = ("AFA", "EAS", "EUR", "HIS", "SAS")
R2_CORE = 0.60
R2_CODING = 0.80
WINDOW = 500_000
MHC = (25_000_000, 34_000_000)
CONTROL_CANDIDATES = 10_000
CONTROL_LD_CANDIDATES = 500
CONTROL_SHORTLIST = 64
CONTROL_RATIO = 4
MAX_CONTROL_OVERLAP = 0.50
PRIMARY_MATCH_COVERAGE = 0.95
FALLBACK_MATCH_COVERAGE = 0.70
MIN_POSITIVE_SIGNALS = 800
MIN_PUBLICATION_LOCI = 400
BASES = {"A", "C", "G", "T"}
COMPLEMENT = str.maketrans("ACGT", "TGCA")
CODING = {
    "transcript_ablation", "splice_acceptor_variant", "splice_donor_variant",
    "stop_gained", "frameshift_variant", "stop_lost", "start_lost",
    "transcript_amplification", "inframe_insertion", "inframe_deletion",
    "missense_variant", "protein_altering_variant", "synonymous_variant",
    "coding_sequence_variant", "incomplete_terminal_codon_variant",
    "start_retained_variant", "stop_retained_variant",
}

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ASTER_DATA_ROOT", ROOT / "data")).resolve()
GWAS = DATA / "gwas"
REF = DATA / "reference"
LD = REF / "1000G_phase3_GRCh37"
STAGE0 = DATA / "shared/stage0"
OUT = DATA / "shared/stage1"
FASTA = REF / "GRCh38_10x_ARC/fasta/genome.fa"
GTF = REF / "GRCh38_10x_ARC/genes/genes.gtf.gz"
CHAIN_FWD = REF / "hg19ToHg38.over.chain.gz"
CHAIN_REV = REF / "hg38ToHg19.over.chain.gz"
BLACKLIST = Path(os.environ.get("ASTER_BLACKLIST", REF / "hg38.blacklist.bed.gz"))
VEP_CACHE = Path(os.environ.get("ASTER_VEP_CACHE", REF / "vep_cache"))
VEP = Path(os.environ.get("ASTER_VEP", shutil.which("vep") or "vep"))
PLINK = os.environ.get("ASTER_PLINK", shutil.which("plink2") or "plink2")
LIFTOVER = os.environ.get("ASTER_LIFTOVER", shutil.which("liftOver") or "liftOver")

SUMSTATS = {
    "MULTI": GWAS / "Suzuki.Nature2024.T2DGGI.MultiAncestry.sumstats.zip",
    "AFA": GWAS / "Suzuki.Nature2024.T2DGGI.AFA.sumstats.zip",
    "EAS": GWAS / "Suzuki.Nature2024.T2DGGI.EAS.sumstats.zip",
    "EUR": GWAS / "EUR_Metal_LDSC-CORR_Neff.v2.txt",
    "HIS": GWAS / "HIS_Metal_LDSC-CORR_Neff.v2.txt",
    "SAS": GWAS / "SAS_Metal_LDSC-CORR_Neff.v2.txt",
}


def run(args: list, **kwargs) -> None:
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def unzip_one(path: Path, target_dir: Path) -> Path:
    if path.suffix != ".zip":
        return path
    with ZipFile(path) as zf:
        member = zf.namelist()[0]
        target = target_dir / Path(member).name
        if not target.exists() or target.stat().st_size != zf.getinfo(member).file_size:
            with zf.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 16 << 20)
    return target


def prepare_sumstats(work: Path) -> dict[str, Path]:
    """Convert each large text table once to chromosome-partitioned parquet."""
    roots: dict[str, Path] = {}
    for name, source in SUMSTATS.items():
        root = work / "sumstats" / name
        roots[name] = root
        if all((root / f"chr{c}.parquet").exists() for c in range(1, 23)):
            continue
        root.mkdir(parents=True, exist_ok=True)
        text = unzip_one(source, work)
        scan = pl.scan_csv(text, separator="\t", infer_schema_length=10_000)
        if name == "MULTI":
            selected = scan.select(
                pl.col("Chromsome").cast(pl.Int16, strict=False).alias("chrom"),
                pl.col("Position").cast(pl.Int32, strict=False).alias("pos37"),
                pl.col("EffectAllele").str.to_uppercase().alias("ea"),
                pl.col("NonEffectAllele").str.to_uppercase().alias("nea"),
                pl.col("PvalAssociation").cast(pl.Float64, strict=False).alias("p_assoc"),
                pl.col("Pval").cast(pl.Float64, strict=False).alias("p_fixed"),
                pl.col("EAF").cast(pl.Float64, strict=False).alias("eaf_multi"),
                pl.col("Neff").cast(pl.Float64, strict=False).alias("neff"),
            )
        else:
            selected = scan.select(
                pl.col("Chromsome").cast(pl.Int16, strict=False).alias("chrom"),
                pl.col("Position").cast(pl.Int32, strict=False).alias("pos37"),
                pl.col("EffectAllele").str.to_uppercase().alias("ea"),
                pl.col("NonEffectAllele").str.to_uppercase().alias("nea"),
                pl.col("Pval").cast(pl.Float64, strict=False).alias(f"p_{name}"),
                pl.col("EAF").cast(pl.Float64, strict=False).alias(f"eaf_{name}"),
                pl.col("Neff").cast(pl.Float64, strict=False).alias(f"neff_{name}"),
            )
        data = (
            selected.filter(
                pl.col("chrom").is_between(1, 22)
                & (pl.col("pos37") > 0)
                & pl.col("ea").is_in(BASES)
                & pl.col("nea").is_in(BASES)
                & (pl.col("ea") != pl.col("nea"))
            )
            .with_columns(
                pl.min_horizontal("ea", "nea").alias("allele_lo"),
                pl.max_horizontal("ea", "nea").alias("allele_hi"),
            )
            .collect(engine="streaming")
            .unique(["chrom", "pos37", "allele_lo", "allele_hi"], keep="first")
        )
        for chrom in range(1, 23):
            data.filter(pl.col("chrom") == chrom).write_parquet(
                root / f"chr{chrom}.parquet", compression="zstd"
            )
        del data
        if source.suffix == ".zip":
            text.unlink(missing_ok=True)
        print(f"Prepared {name}", flush=True)
    return roots


def load_gwas(chrom: int, roots: dict[str, Path]) -> pl.DataFrame:
    data = pl.read_parquet(roots["MULTI"] / f"chr{chrom}.parquet")
    keys = ["chrom", "pos37", "allele_lo", "allele_hi"]
    for name in ANCESTRIES:
        part = pl.read_parquet(roots[name] / f"chr{chrom}.parquet").drop("ea", "nea")
        data = data.join(part, on=keys, how="left")
    return data.with_columns(
        pl.concat_str(
            pl.col("chrom"), pl.col("pos37"), pl.col("allele_lo"), pl.col("allele_hi"),
            separator=":"
        ).alias("key37")
    )


def make_pgen(chrom: int, pop: str, chrom_work: Path) -> tuple[Path, pl.DataFrame]:
    vcf = LD / f"vcf/ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    full = chrom_work / f"chr{chrom}_ALL"
    if not Path(f"{full}.pgen").exists():
        run([
            PLINK, "--vcf", vcf, "--double-id", "--snps-only", "just-acgt",
            "--max-alleles", "2", "--set-all-var-ids", "@:#:$r:$a",
            "--new-id-max-allele-len", "10", "missing", "--make-pgen",
            "--threads", THREADS, "--out", full,
        ], stdout=subprocess.DEVNULL)
    out = chrom_work / f"chr{chrom}_{pop}"
    run([
        PLINK, "--pfile", full, "--keep", STAGE0 / f"ld_samples/{pop}.keep",
        "--maf", "0.005", "--mac", "10", "--geno", "0.05", "--make-pgen",
        "--threads", THREADS, "--out", out,
    ], stdout=subprocess.DEVNULL)
    run([PLINK, "--pfile", out, "--freq", "counts", "--out", out], stdout=subprocess.DEVNULL)
    pvar = pl.read_csv(Path(f"{out}.pvar"), separator="\t", comment_prefix="##").select(
        pl.col("#CHROM").cast(pl.Int16).alias("chrom"),
        pl.col("POS").cast(pl.Int32).alias("pos37"),
        pl.col("ID").alias("plink_id"),
        pl.col("REF").alias("ref37"),
        pl.col("ALT").alias("alt37"),
    ).with_columns(
        pl.min_horizontal("ref37", "alt37").alias("allele_lo"),
        pl.max_horizontal("ref37", "alt37").alias("allele_hi"),
    )
    counts = pl.read_csv(Path(f"{out}.acount"), separator="\t").select(
        pl.col("ID").alias("plink_id"),
        pl.col("ALT_CTS").cast(pl.Float64),
        pl.col("OBS_CT").cast(pl.Float64),
    ).with_columns(
        (pl.min_horizontal("ALT_CTS", pl.col("OBS_CT") - pl.col("ALT_CTS")) / pl.col("OBS_CT"))
        .alias(f"maf_{pop}")
    ).select("plink_id", f"maf_{pop}")
    return out, pvar.join(counts, on="plink_id", how="left")


def ld_for_anchors(prefix: Path, anchor_ids: list[str], tag: str) -> pl.DataFrame:
    columns = {
        "ID_A": pl.String, "ID_B": pl.String, "POS_A": pl.Int32, "POS_B": pl.Int32,
        "NONMAJ_FREQ_A": pl.Float64, "NONMAJ_FREQ_B": pl.Float64,
        "UNPHASED_R2": pl.Float64,
    }
    if not anchor_ids:
        return pl.DataFrame(schema=columns)
    ids = prefix.parent / f"{tag}.anchors.txt"
    ids.write_text("".join(f"{x}\n" for x in sorted(set(anchor_ids))))
    out = prefix.parent / tag
    run([
        PLINK, "--pfile", prefix, "--r2-unphased", "cols=pos,id,freq",
        "--ld-snp-list", ids, "--ld-window-kb", "500", "--ld-window", "99999999",
        "--ld-window-r2", str(R2_CORE), "--threads", THREADS, "--out", out,
    ], stdout=subprocess.DEVNULL)
    return pl.read_csv(Path(f"{out}.vcor"), separator="\t", schema_overrides=columns)


def far_from(values: np.ndarray, forbidden: np.ndarray, distance: int) -> np.ndarray:
    if forbidden.size == 0:
        return np.ones(values.size, dtype=bool)
    forbidden = np.sort(forbidden)
    idx = np.searchsorted(forbidden, values)
    left = np.full(values.size, np.inf)
    right = np.full(values.size, np.inf)
    mask = idx > 0
    left[mask] = values[mask] - forbidden[idx[mask] - 1]
    mask = idx < forbidden.size
    right[mask] = forbidden[idx[mask]] - values[mask]
    return np.minimum(left, right) > distance


def finite_array(values) -> np.ndarray:
    return np.asarray([np.nan if x is None else x for x in values], dtype=float)


def finite_mean(values) -> float:
    array = finite_array(values)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def anchor_shortlists(
    gwas: pl.DataFrame,
    panel: pl.DataFrame,
    signals: pl.DataFrame,
    loci: pl.DataFrame,
) -> tuple[dict[str, list[str]], pl.DataFrame]:
    keys = ["chrom", "pos37", "allele_lo", "allele_hi"]
    pool = gwas.join(panel, on=keys, how="inner")
    pcols = [f"p_{x}" for x in ANCESTRIES]
    pool = pool.with_columns(
        pl.sum_horizontal([pl.col(x).is_not_null().cast(pl.Int8) for x in pcols]).alias("n_ancestry"),
        pl.min_horizontal([pl.col(x).fill_null(1.0) for x in pcols]).alias("min_ancestry_p"),
        pl.mean_horizontal([pl.col(f"maf_{x}") for x in POPS]).alias("maf_mean"),
        pl.sum_horizontal([pl.col(f"maf_{x}").is_not_null().cast(pl.Int8) for x in POPS]).alias("n_panels"),
    )
    neff_gate = float(pool["neff"].quantile(0.90)) * 0.50
    pool = pool.filter(
        (pl.col("p_assoc") >= 0.05) & (pl.col("p_fixed") >= 0.05)
        & (pl.col("n_ancestry") >= 3) & (pl.col("min_ancestry_p") >= 1e-4)
        & (pl.col("neff") >= neff_gate)
    )
    positions = pool["pos37"].to_numpy()
    keep = far_from(positions, gwas.filter(pl.col("p_assoc") <= 1e-4)["pos37"].to_numpy(), WINDOW)
    for row in loci.iter_rows(named=True):
        keep &= ~((positions >= row["start37"] - 1_000_000) & (positions <= row["end37"] + 1_000_000))
    pool = pool.filter(pl.Series("keep", keep))

    signal_anchors = signals.join(panel, on=keys, how="inner").with_columns(
        pl.mean_horizontal([pl.col(f"maf_{x}") for x in POPS]).alias("maf_mean"),
        pl.sum_horizontal([pl.col(f"maf_{x}").is_not_null().cast(pl.Int8) for x in POPS]).alias("n_panels"),
    )
    shortlists: dict[str, list[str]] = {}
    panel_positions = np.sort(panel["pos37"].to_numpy())
    pool_positions = pool["pos37"].to_numpy()
    pool_density = (
        np.searchsorted(panel_positions, pool_positions + WINDOW, side="right")
        - np.searchsorted(panel_positions, pool_positions - WINDOW)
    )
    pool = pool.with_columns(pl.Series("local_variant_density", pool_density))
    p = pool.to_dict(as_series=False)
    for signal in signal_anchors.iter_rows(named=True):
        shared_max = np.zeros(pool.height)
        shared_n = np.zeros(pool.height)
        for pop in POPS:
            candidate = finite_array(p[f"maf_{pop}"])
            target = signal.get(f"maf_{pop}")
            if target is None or not math.isfinite(target):
                continue
            valid = np.isfinite(candidate)
            shared_n += valid
            shared_max = np.maximum(shared_max, np.where(valid, np.abs(candidate - target), 0.0))
        neff_diff = np.abs(np.log(finite_array(p["neff"]) / signal["neff"]))
        mean_diff = np.abs(finite_array(p["maf_mean"]) - signal["maf_mean"])
        count_diff = np.abs(finite_array(p["n_panels"]) - signal["n_panels"])
        target_density = (
            np.searchsorted(panel_positions, signal["pos37"] + WINDOW, side="right")
            - np.searchsorted(panel_positions, signal["pos37"] - WINDOW)
        )
        density_diff = np.abs(
            np.log1p(finite_array(p["local_variant_density"])) - math.log1p(target_density)
        )
        feasible = (
            (shared_n > 0) & (shared_max <= 0.10) & (mean_diff <= 0.05)
            & (neff_diff <= 0.05) & (count_diff <= 1) & (density_diff <= 0.70)
        )
        idx = np.flatnonzero(feasible)
        if idx.size:
            score = (
                5 * mean_diff[idx] + neff_diff[idx] + 0.2 * count_diff[idx]
                + 0.5 * density_diff[idx] + 0.02 * shared_max[idx]
            )
            idx = idx[np.argsort(score, kind="stable")[:CONTROL_CANDIDATES]]
            shortlists[signal["official_signal_id"]] = [p["key37"][i] for i in idx]
    return shortlists, pool


def context_match_shortlists(
    shortlists: dict[str, list[str]],
    signals: pl.DataFrame,
    chrom_work: Path,
    tss: dict[str, np.ndarray],
) -> dict[str, list[str]]:
    signal_keys = dict(zip(signals["official_signal_id"], signals["key37"]))
    keys = sorted(
        set(signal_keys.values()) | {key for values in shortlists.values() for key in values},
        key=lambda key: int(key.split(":")[1]),
    )
    bed37 = chrom_work / "control_context_b37.bed"
    with bed37.open("w") as handle:
        for key in keys:
            chrom, pos, *_ = key.split(":")
            handle.write(f"chr{chrom}\t{int(pos)-1}\t{pos}\t{key}\n")
    bed38 = chrom_work / "control_context_b38.bed"
    unmapped = chrom_work / "control_context_unmapped.bed"
    run([LIFTOVER, bed37, CHAIN_FWD, bed38, unmapped], stdout=subprocess.DEVNULL)
    mapped: dict[str, tuple[str, int]] = {}
    with bed38.open() as handle:
        for line in handle:
            chrom, start, _, key = line.rstrip().split("\t")[:4]
            if key not in mapped:
                mapped[key] = (chrom, int(start) + 1)
            else:
                mapped[key] = ("", -1)
    fasta = pysam.FastaFile(str(FASTA))
    features = {}
    for key, (chrom, pos) in mapped.items():
        if chrom not in fasta.references or pos <= 2048 or pos + 2048 > fasta.get_reference_length(chrom):
            continue
        distance, density = gene_features(chrom, pos, tss)
        features[key] = np.array([
            math.log1p(distance),
            float(density),
            gc_fraction({"chr38": chrom, "pos38": pos}, fasta),
        ])
    reranked = {}
    for sid, candidates in shortlists.items():
        target = features.get(signal_keys.get(sid, ""))
        if target is None:
            continue
        available = [key for key in candidates if key in features]
        available.sort(
            key=lambda key: (
                abs(features[key][0] - target[0])
                + abs(features[key][1] - target[1]) / 2
                + abs(features[key][2] - target[2]) / 0.02,
                key,
            )
        )
        reranked[sid] = available[:CONTROL_LD_CANDIDATES]
    return reranked


def chromosome_bags(
    chrom: int,
    roots: dict[str, Path],
    signals_all: pl.DataFrame,
    loci_all: pl.DataFrame,
    work: Path,
    tss_match: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict], dict[str, dict], dict[str, dict]]:
    chrom_work = work / f"chr{chrom}"
    chrom_work.mkdir(parents=True, exist_ok=True)
    gwas = load_gwas(chrom, roots)
    signals = signals_all.filter(pl.col("chrom") == chrom).with_columns(
        pl.min_horizontal("risk_allele", "other_allele").alias("allele_lo"),
        pl.max_horizontal("risk_allele", "other_allele").alias("allele_hi"),
    )
    loci = loci_all.filter(pl.col("chrom") == chrom)

    prefixes: dict[str, Path] = {}
    pop_tables: dict[str, pl.DataFrame] = {}
    for pop in POPS:
        prefixes[pop], pop_tables[pop] = make_pgen(chrom, pop, chrom_work)

    keys = ["chrom", "pos37", "allele_lo", "allele_hi"]
    panel = pl.concat(
        [pop_tables[pop].select(keys + ["ref37", "alt37"]) for pop in POPS],
        how="vertical",
    ).unique(keys)
    for pop in POPS:
        panel = panel.join(
            pop_tables[pop].select(keys + [f"maf_{pop}"]),
            on=keys,
            how="left",
        )
    panel = panel.with_columns(
        pl.concat_str(pl.col("chrom"), pl.col("pos37"), pl.col("allele_lo"), pl.col("allele_hi"), separator=":").alias("key37")
    )
    signal_panel = signals.join(panel, on=keys, how="inner")
    signal_by_plink: dict[str, dict[str, str]] = {pop: {} for pop in POPS}
    for pop in POPS:
        mapping = signals.join(pop_tables[pop], on=keys, how="inner")
        signal_by_plink[pop] = dict(zip(mapping["plink_id"], mapping["official_signal_id"]))

    r2_pos: dict[str, dict[str, float]] = {x: {} for x in signals["official_signal_id"]}
    for pop in POPS:
        ids = list(signal_by_plink[pop])
        ld = ld_for_anchors(prefixes[pop], ids, f"positive_{pop}")
        id_to_key = dict(zip(pop_tables[pop]["plink_id"], pop_tables[pop].with_columns(
            pl.concat_str(pl.col("chrom"), pl.col("pos37"), pl.col("allele_lo"), pl.col("allele_hi"), separator=":").alias("key37")
        )["key37"]))
        for anchor_id, sid in signal_by_plink[pop].items():
            r2_pos[sid][id_to_key[anchor_id]] = 1.0
        for row in ld.iter_rows(named=True):
            sid = signal_by_plink[pop].get(row["ID_A"])
            key = id_to_key.get(row["ID_B"])
            if sid and key:
                r2_pos[sid][key] = max(r2_pos[sid].get(key, 0.0), float(row["UNPHASED_R2"]))

    tested = set(gwas["key37"])
    signal_rows = {x["official_signal_id"]: x for x in signal_panel.iter_rows(named=True)}
    positives: list[dict] = []
    for sid, members in r2_pos.items():
        if sid not in signal_rows:
            continue
        row = signal_rows[sid]
        members = {
            key: r2 for key, r2 in members.items()
            if key in tested and abs(int(key.split(":")[1]) - row["pos37"]) <= WINDOW
        }
        if members:
            anchor_key = f"{chrom}:{row['pos37']}:{min(row['risk_allele'], row['other_allele'])}:{max(row['risk_allele'], row['other_allele'])}"
            positives.append({"signal": row, "anchor": anchor_key, "members": members})

    shortlists, null_pool = anchor_shortlists(gwas, panel, signals, loci)
    shortlists = context_match_shortlists(
        shortlists, signal_panel, chrom_work, tss_match
    )
    selected_anchor_keys = sorted({key for values in shortlists.values() for key in values})
    pool_rows = {x["key37"]: x for x in null_pool.filter(pl.col("key37").is_in(selected_anchor_keys)).iter_rows(named=True)}
    r2_ctrl: dict[str, dict[str, float]] = {key: {} for key in selected_anchor_keys}
    for pop in POPS:
        table = pop_tables[pop].with_columns(
            pl.concat_str(pl.col("chrom"), pl.col("pos37"), pl.col("allele_lo"), pl.col("allele_hi"), separator=":").alias("key37")
        )
        key_to_id = dict(zip(table["key37"], table["plink_id"]))
        id_to_key = dict(zip(table["plink_id"], table["key37"]))
        ids = [key_to_id[key] for key in selected_anchor_keys if key in key_to_id]
        ld = ld_for_anchors(prefixes[pop], ids, f"controls_{pop}")
        for key in selected_anchor_keys:
            if key in key_to_id:
                r2_ctrl[key][key] = 1.0
        for row in ld.iter_rows(named=True):
            anchor = id_to_key.get(row["ID_A"])
            member = id_to_key.get(row["ID_B"])
            if anchor and member:
                r2_ctrl[anchor][member] = max(r2_ctrl[anchor].get(member, 0.0), float(row["UNPHASED_R2"]))

    controls: list[dict] = []
    positive_sizes = {x["signal"]["official_signal_id"]: len(x["members"]) for x in positives}
    for sid, anchors in shortlists.items():
        if sid not in positive_sizes:
            continue
        ranked = []
        for context_rank, anchor in enumerate(anchors):
            members = {key: r2 for key, r2 in r2_ctrl.get(anchor, {}).items() if key in tested}
            if len(members) >= positive_sizes[sid]:
                span = max(int(x.split(":")[1]) for x in members) - min(int(x.split(":")[1]) for x in members)
                size_diff = abs(math.log1p(len(members)) - math.log1p(positive_sizes[sid]))
                ranked.append((context_rank, size_diff, span, anchor, members))
        for _, _, _, anchor, members in sorted(ranked)[:CONTROL_SHORTLIST]:
            controls.append({"target_signal": sid, "anchor": anchor, "members": members, "anchor_row": pool_rows[anchor]})

    needed = (
        {key for bag in positives for key in bag["members"]}
        | {key for bag in controls for key in bag["members"]}
    )
    metrics = {
        row["key37"]: row
        for row in panel.iter_rows(named=True)
        if row["key37"] in needed
    }
    anchors = {x["anchor"]: x["signal"] for x in positives}
    anchors.update({x["anchor"]: x["anchor_row"] for x in controls})
    print(f"chr{chrom}: {len(positives)} positive raw bags, {len(controls)} control candidates", flush=True)
    shutil.rmtree(chrom_work)
    return positives, controls, metrics, anchors


def load_intervals(path: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                chrom, start, end, *_ = line.rstrip().split("\t")
                out[chrom].append((int(start), int(end)))
    for chrom in out:
        out[chrom].sort()
    return out


def overlaps_interval(
    chrom: str, start: int, end: int, intervals: dict[str, list[tuple[int, int]]]
) -> bool:
    blocks = intervals.get(chrom, ())
    index = bisect_left(blocks, (end, -1))
    return any(block_end > start for _, block_end in blocks[max(0, index - 2):index])


def liftover_variants(keys: set[str], metrics: dict[str, dict], work: Path) -> dict[str, dict]:
    bed37 = work / "variants_b37.bed"
    with bed37.open("w") as handle:
        for key in sorted(keys, key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1]), x)):
            chrom, pos, *_ = key.split(":")
            handle.write(f"chr{chrom}\t{int(pos)-1}\t{pos}\t{key}\n")
    bed38, unmapped = work / "variants_b38.bed", work / "variants_b38.unmapped"
    run([LIFTOVER, bed37, CHAIN_FWD, bed38, unmapped], stdout=subprocess.DEVNULL)
    mapped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with bed38.open() as handle:
        for line in handle:
            chrom, start, _, key = line.rstrip().split("\t")[:4]
            mapped[key].append((chrom, int(start)))

    fasta = pysam.FastaFile(str(FASTA))
    blacklist = load_intervals(BLACKLIST)
    eligible: dict[str, dict] = {}
    for key, hits in mapped.items():
        if len(hits) != 1:
            continue
        chrom38, pos0 = hits[0]
        if chrom38 not in fasta.references or pos0 < 10_000 or pos0 + 10_001 > fasta.get_reference_length(chrom38):
            continue
        lo, hi = key.split(":")[2:]
        ref38 = fasta.fetch(chrom38, pos0, pos0 + 1).upper()
        alleles = (lo, hi)
        if ref38 not in alleles:
            alleles = tuple(x.translate(COMPLEMENT) for x in alleles)
        if ref38 not in alleles or overlaps_interval(chrom38, pos0 - 10_000, pos0 + 10_001, blacklist):
            continue
        alt38 = alleles[1] if alleles[0] == ref38 else alleles[0]
        key38 = f"{chrom38}:{pos0+1}:{ref38}:{alt38}"
        eligible[key] = {
            **metrics[key], "chr38": chrom38, "pos38": pos0 + 1,
            "ref": ref38, "alt": alt38, "key38": key38,
        }

    bed38_round = work / "roundtrip_input.bed"
    with bed38_round.open("w") as handle:
        for key37, row in eligible.items():
            handle.write(f"{row['chr38']}\t{row['pos38']-1}\t{row['pos38']}\t{key37}\n")
    returned, failed = work / "roundtrip_b37.bed", work / "roundtrip_failed.bed"
    run([LIFTOVER, bed38_round, CHAIN_REV, returned, failed], stdout=subprocess.DEVNULL)
    roundtrip = {}
    with returned.open() as handle:
        for line in handle:
            chrom, start, _, key = line.rstrip().split("\t")[:4]
            roundtrip[key] = (chrom.removeprefix("chr"), int(start) + 1)
    return {
        key: row for key, row in eligible.items()
        if roundtrip.get(key) == (key.split(":")[0], int(key.split(":")[1]))
    }


def vep_coding(variants: dict[str, dict], work: Path) -> set[str]:
    vcf = work / "variants_b38.vcf"
    with vcf.open("w") as handle:
        handle.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for key37, row in sorted(variants.items(), key=lambda x: (x[1]["chr38"], x[1]["pos38"], x[1]["key38"])):
            handle.write(f"{row['chr38']}\t{row['pos38']}\t{row['key38']}\t{row['ref']}\t{row['alt']}\t.\tPASS\t.\n")
    output = work / "vep.tsv"
    digest = hashlib.sha256(vcf.read_bytes()).hexdigest()
    stamp = work / "vep_input.sha256"
    cached_digest = stamp.read_text().strip() if stamp.exists() else ""
    if not output.exists() or output.stat().st_size == 0 or cached_digest != digest:
        run([
            VEP, "--input_file", vcf, "--output_file", output, "--format", "vcf", "--tab",
            "--fields", "Uploaded_variation,Consequence,BIOTYPE", "--offline", "--cache",
            "--dir_cache", VEP_CACHE, "--assembly", "GRCh38", "--fork", "16",
            "--no_stats", "--force_overwrite",
        ])
        stamp.write_text(digest + "\n")
    coding38: set[str] = set()
    with output.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            variant, consequence, biotype = line.rstrip().split("\t")[:3]
            if biotype == "protein_coding" and CODING.intersection(consequence.split(",")):
                coding38.add(variant)
    return {key37 for key37, row in variants.items() if row["key38"] in coding38}


def parse_genes() -> dict[str, np.ndarray]:
    tss: dict[str, list[int]] = defaultdict(list)
    with gzip.open(GTF, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            chrom, _, feature, start, end, _, strand, _, attr = line.rstrip().split("\t")
            if feature == "gene" and 'gene_type "protein_coding"' in attr:
                tss[chrom].append(int(start) if strand == "+" else int(end))
    return {chrom: np.sort(values) for chrom, values in tss.items()}


def gene_features(chrom: str, pos: int, tss: dict[str, np.ndarray]) -> tuple[float, int]:
    values = tss.get(chrom, np.array([], dtype=int))
    if values.size == 0:
        return 1e9, 0
    idx = np.searchsorted(values, pos)
    near = []
    if idx:
        near.append(abs(pos - values[idx - 1]))
    if idx < values.size:
        near.append(abs(values[idx] - pos))
    density = np.searchsorted(values, pos + WINDOW, side="right") - np.searchsorted(values, pos - WINDOW)
    return float(min(near)), int(density)


def gc_fraction(row: dict, fasta: pysam.FastaFile) -> float:
    pos0 = row["pos38"] - 1
    seq = fasta.fetch(row["chr38"], pos0 - 2048, pos0 + 2048).upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def clean_bags(
    positives: list[dict], controls: list[dict], variants: dict[str, dict], coding: set[str]
) -> tuple[list[dict], list[dict]]:
    def clean(bag: dict) -> dict | None:
        if bag["anchor"] in coding or any(key in coding and r2 >= R2_CODING for key, r2 in bag["members"].items()):
            return None
        members = {key: r2 for key, r2 in bag["members"].items() if key in variants and key not in coding}
        if not members:
            return None
        return {**bag, "members": members}

    positive_clean = []
    for bag in positives:
        cleaned = clean(bag)
        if cleaned is not None:
            positive_clean.append(cleaned)
    valid_signals = {x["signal"]["official_signal_id"] for x in positive_clean}
    control_clean = []
    for bag in controls:
        if bag["target_signal"] in valid_signals:
            cleaned = clean(bag)
            if cleaned is not None:
                control_clean.append(cleaned)
    return positive_clean, control_clean


def variant_maf(key: str, variants: dict[str, dict], cache: dict[str, float]) -> float:
    if key not in cache:
        cache[key] = finite_mean(variants[key].get(f"maf_{pop}") for pop in POPS)
    return cache[key]


def exact_members(
    target: dict,
    candidate: dict,
    variants: dict[str, dict],
    maf_cache: dict[str, float],
) -> list[str] | None:
    m = len(target["members"])
    if len(candidate["members"]) < m:
        return None
    target_anchor = int(target["anchor"].split(":")[1])
    candidate_anchor = int(candidate["anchor"].split(":")[1])
    candidate_keys = sorted(candidate["members"])
    candidate_r2 = np.array([candidate["members"][key] for key in candidate_keys])
    candidate_distance = np.array(
        [abs(int(key.split(":")[1]) - candidate_anchor) / WINDOW for key in candidate_keys]
    )
    candidate_maf = np.array([variant_maf(key, variants, maf_cache) for key in candidate_keys])
    available = np.ones(len(candidate_keys), dtype=bool)
    chosen: list[str] = []
    targets = sorted(target["members"], key=lambda k: (-target["members"][k], abs(int(k.split(":")[1]) - target_anchor)))
    for key in targets:
        r2 = target["members"][key]
        distance = abs(int(key.split(":")[1]) - target_anchor) / WINDOW
        maf = variant_maf(key, variants, maf_cache)
        cost = (
            np.abs(candidate_r2 - r2)
            + np.abs(candidate_distance - distance)
            + np.abs(candidate_maf - maf)
        )
        cost[~available] = np.inf
        best = int(np.argmin(cost))
        chosen.append(candidate_keys[best])
        available[best] = False
    return chosen


def bag_features(
    bag: dict,
    members: list[str],
    variants: dict[str, dict],
    anchors: dict[str, dict],
    tss,
    fasta,
    gc_cache: dict[str, float],
) -> dict:
    anchor = variants[bag["anchor"]]
    anchor_meta = anchors[bag["anchor"]]
    positions = [variants[x]["pos38"] for x in members]
    tss_distance, gene_density = gene_features(anchor["chr38"], anchor["pos38"], tss)
    for key in members:
        if key not in gc_cache:
            gc_cache[key] = gc_fraction(variants[key], fasta)
    values = {
        "maf_mean": finite_mean(anchor.get(f"maf_{p}") for p in POPS),
        "log_neff": math.log(float(anchor_meta["neff"])),
        "log_ld_buddies": math.log1p(len(members)),
        "log_span": math.log1p(max(positions) - min(positions)),
        "log_tss": math.log1p(tss_distance),
        "gene_density": float(gene_density),
        "gc": float(np.mean([gc_cache[key] for key in members])),
    }
    return values


def match_controls(positives, controls, variants, anchors):
    controls_by_signal: dict[str, list[dict]] = defaultdict(list)
    for bag in controls:
        controls_by_signal[bag["target_signal"]].append(bag)
    tss = parse_genes()
    fasta = pysam.FastaFile(str(FASTA))
    maf_cache: dict[str, float] = {}
    gc_cache: dict[str, float] = {}
    edges: list[dict] = []
    pos_features: dict[str, dict] = {}
    for positive in positives:
        sid = positive["signal"]["official_signal_id"]
        members = list(positive["members"])
        pos_features[sid] = bag_features(
            positive, members, variants, anchors, tss, fasta, gc_cache
        )
        for candidate in controls_by_signal.get(sid, ()):
            chosen = exact_members(positive, candidate, variants, maf_cache)
            if chosen is None:
                continue
            features = bag_features(
                candidate, chosen, variants, anchors, tss, fasta, gc_cache
            )
            edges.append({"signal": sid, "anchor": candidate["anchor"], "bag": candidate, "members": chosen, "features": features})
    feature_names = list(next(iter(pos_features.values())))
    pooled = np.array([[edge["features"][x] for x in feature_names] for edge in edges] + [[x[name] for name in feature_names] for x in pos_features.values()])
    scale = np.nanstd(pooled, axis=0)
    scale[scale == 0] = 1
    for edge in edges:
        target = pos_features[edge["signal"]]
        edge["cost"] = float(np.linalg.norm(
            (np.array([edge["features"][x] for x in feature_names]) - np.array([target[x] for x in feature_names])) / scale
        ))

    edge_by_signal: dict[str, list[int]] = defaultdict(list)
    edge_by_anchor: dict[str, list[int]] = defaultdict(list)
    for i, edge in enumerate(edges):
        edge_by_signal[edge["signal"]].append(i)
        edge_by_anchor[edge["anchor"]].append(i)
    edge_members = [set(edge["members"]) for edge in edges]
    inverted: dict[str, list[int]] = defaultdict(list)
    for edge_id, members in enumerate(edge_members):
        for key in members:
            inverted[key].append(edge_id)
    shared: dict[tuple[int, int], int] = defaultdict(int)
    for edge_ids in inverted.values():
        for i, left in enumerate(edge_ids):
            for right in edge_ids[i + 1:]:
                if edges[left]["anchor"] != edges[right]["anchor"]:
                    shared[(min(left, right), max(left, right))] += 1
    conflicts = {
        pair for pair, count in shared.items()
        if count / min(len(edge_members[pair[0]]), len(edge_members[pair[1]]))
        >= MAX_CONTROL_OVERLAP
    }

    def solve(ratio: int, required_coverage: float):
        eligible = sorted(
            sid for sid in pos_features if len(edge_by_signal.get(sid, ())) >= ratio
        )
        coverage = len(eligible) / len(pos_features)
        print(
            f"Control eligibility 1:{ratio}: {len(eligible)}/{len(pos_features)} "
            f"clean positives ({coverage:.1%})",
            flush=True,
        )
        if coverage < required_coverage:
            return None
        n_edges = len(edges)
        z_index = {sid: n_edges + i for i, sid in enumerate(eligible)}
        n_variables = n_edges + len(eligible)
        rows, lower, upper = [], [], []
        for sid in eligible:
            idx = edge_by_signal[sid]
            rows.append((idx + [z_index[sid]], [1.0] * len(idx) + [-ratio]))
            lower.append(0); upper.append(0)
        for idx in edge_by_anchor.values():
            rows.append((idx, [1.0] * len(idx))); lower.append(0); upper.append(1)
        for left, right in conflicts:
            rows.append(([left, right], [1.0, 1.0])); lower.append(0); upper.append(1)
        rr, cc, vv = [], [], []
        for row_id, (idx, values) in enumerate(rows):
            rr.extend([row_id] * len(idx)); cc.extend(idx); vv.extend(values)
        matrix = coo_array((vv, (rr, cc)), shape=(len(rows), n_variables)).tocsr()
        constraints = LinearConstraint(matrix, np.array(lower), np.array(upper))
        upper_bounds = np.ones(n_variables)
        eligible_set = set(eligible)
        for i, edge in enumerate(edges):
            if edge["signal"] not in eligible_set:
                upper_bounds[i] = 0
        bounds = Bounds(np.zeros(n_variables), upper_bounds)
        integrality = np.ones(n_variables)
        cardinality_objective = np.zeros(n_variables)
        cardinality_objective[n_edges:] = -1
        cardinality = milp(
            c=cardinality_objective, integrality=integrality, bounds=bounds,
            constraints=constraints,
            options={"time_limit": 7200, "mip_rel_gap": 0.0},
        )
        if not cardinality.success:
            print(f"Maximum-cardinality MILP failed: {cardinality.message}", flush=True)
            return None
        maximum = int(round(cardinality.x[n_edges:].sum()))
        required = math.ceil(required_coverage * len(pos_features))
        print(
            f"Maximum feasible 1:{ratio} coverage: {maximum}/{len(pos_features)} "
            f"({maximum / len(pos_features):.1%})",
            flush=True,
        )
        if maximum < required:
            return None
        cardinality_row = coo_array(
            (
                np.ones(len(eligible)),
                (np.zeros(len(eligible), dtype=np.int32), list(z_index.values())),
            ),
            shape=(1, n_variables),
        ).tocsr()
        target_cardinality = required
        distance_objective = np.zeros(n_variables)
        distance_objective[:n_edges] = [edge["cost"] for edge in edges]
        result = milp(
            c=distance_objective, integrality=integrality, bounds=bounds,
            constraints=[
                constraints,
                LinearConstraint(cardinality_row, target_cardinality, target_cardinality),
            ],
            options={"time_limit": 7200, "mip_rel_gap": 0.0},
        )
        if not result.success:
            print(f"Minimum-distance MILP failed: {result.message}", flush=True)
            return None
        matched = {
            sid for sid, index in z_index.items() if result.x[index] > 0.5
        }
        return result, matched

    ratio = CONTROL_RATIO
    solved = solve(ratio, PRIMARY_MATCH_COVERAGE)
    if solved is None or not solved[0].success:
        ratio = 2
        solved = solve(ratio, FALLBACK_MATCH_COVERAGE)
    if solved is None or not solved[0].success:
        raise RuntimeError("Control matching is infeasible at both 1:4 and 1:2")
    result, matched = solved
    selected = [edges[i] for i in np.flatnonzero(result.x[:len(edges)] > 0.5)]
    if {edge["signal"] for edge in selected} != matched:
        raise RuntimeError("MILP edge and signal selections disagree")
    return selected, {sid: pos_features[sid] for sid in sorted(matched)}, ratio, feature_names


def validate_matching(selected, pos_features, feature_names, folds_by_chrom):
    positive = np.array([[row[name] for name in feature_names] for row in pos_features.values()])
    negative = np.array([[edge["features"][name] for name in feature_names] for edge in selected])
    smd = np.abs((positive.mean(0) - negative.mean(0)) / np.sqrt((positive.var(0, ddof=1) + negative.var(0, ddof=1)) / 2))
    ks = np.array([ks_2samp(positive[:, i], negative[:, i]).statistic for i in range(positive.shape[1])])
    print("Matching:", ", ".join(f"{name}:SMD={smd[i]:.3f},KS={ks[i]:.3f}" for i, name in enumerate(feature_names)), flush=True)
    technical = [feature_names.index(name) for name in ("maf_mean", "log_neff", "log_ld_buddies", "log_span")]
    biological = [feature_names.index(name) for name in ("log_tss", "gene_density", "gc")]
    if (
        np.nanmax(smd[technical]) >= 0.10
        or np.nanmax(smd[biological]) >= 0.25
        or np.nanmax(ks) >= 0.25
    ):
        raise RuntimeError("Matched controls fail SMD/KS release threshold")


def assign_folds(positives: list[dict], selected: list[dict]) -> dict[int, int]:
    pos_counts = defaultdict(int)
    variants = defaultdict(int)
    for bag in positives:
        chrom = int(bag["anchor"].split(":")[0])
        pos_counts[chrom] += 1
        variants[chrom] += len(bag["members"])
    for edge in selected:
        chrom = int(edge["anchor"].split(":")[0])
        variants[chrom] += len(edge["members"])
    loads = [[0, 0] for _ in range(5)]
    assignment = {}
    for chrom in sorted(pos_counts, key=lambda c: (pos_counts[c], variants[c]), reverse=True):
        fold = min(range(5), key=lambda x: (loads[x][0], loads[x][1], x))
        assignment[chrom] = fold
        loads[fold][0] += pos_counts[chrom]
        loads[fold][1] += variants[chrom]
    return assignment


def nuisance_check(rows: list[dict], feature_names: list[str]) -> float:
    x = np.array([[row["features"][name] for name in feature_names] for row in rows])
    y = np.array([row["y"] for row in rows])
    folds = np.array([row["fold"] for row in rows])
    pred = np.zeros(len(rows))
    for fold in sorted(set(folds)):
        train, test = folds != fold, folds == fold
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(x[train], y[train])
        pred[test] = model.predict_proba(x[test])[:, 1]
    auc = roc_auc_score(y, pred)
    print(f"Nuisance-only outer-fold AUROC={auc:.3f}", flush=True)
    if auc > 0.60:
        raise RuntimeError("Nuisance-only AUROC exceeds 0.60")
    return auc


def save_outputs(positives, selected, variants, pos_features, ratio, feature_names):
    selected_by_signal: dict[str, list[dict]] = defaultdict(list)
    for edge in selected:
        selected_by_signal[edge["signal"]].append(edge)
    positives = [bag for bag in positives if bag["signal"]["official_signal_id"] in selected_by_signal]
    n_loci = len({bag["signal"]["official_locus_id"] for bag in positives})
    if len(positives) < MIN_POSITIVE_SIGNALS or n_loci < MIN_PUBLICATION_LOCI:
        raise RuntimeError(
            f"Insufficient release scale: {len(positives)} signals and {n_loci} publication loci"
        )
    fold_by_chrom = assign_folds(positives, selected)
    clean_per_locus = defaultdict(int)
    for bag in positives:
        clean_per_locus[bag["signal"]["official_locus_id"]] += 1

    bags: list[dict] = []
    metadata: list[dict] = []
    nuisance_rows: list[dict] = []
    for positive in positives:
        signal = positive["signal"]
        sid = signal["official_signal_id"]
        chrom = int(positive["anchor"].split(":")[0])
        fold = fold_by_chrom[chrom]
        group_weight = 1.0 / clean_per_locus[signal["official_locus_id"]]
        pos_id = f"MIL_POS_{sid}"
        bags.append({"mil_locus_id": pos_id, "members": list(positive["members"])})
        metadata.append({
            "mil_locus_id": pos_id, "y": 1, "match_set_id": sid,
            "dependency_group": signal["official_locus_id"], "group_weight": group_weight,
            "official_signal_id": sid, "official_locus_id": signal["official_locus_id"],
            "anchor_key37": positive["anchor"], "fold": fold,
        })
        nuisance_rows.append({"y": 1, "fold": fold, "features": pos_features[sid]})
        for order, edge in enumerate(sorted(selected_by_signal[sid], key=lambda x: x["anchor"]), 1):
            ctrl_id = f"MIL_CTRL_{sid}_{order}"
            bags.append({"mil_locus_id": ctrl_id, "members": edge["members"]})
            metadata.append({
                "mil_locus_id": ctrl_id, "y": 0, "match_set_id": sid,
                "dependency_group": signal["official_locus_id"], "group_weight": group_weight,
                "official_signal_id": "", "official_locus_id": signal["official_locus_id"],
                "anchor_key37": edge["anchor"], "fold": fold,
            })
            nuisance_rows.append({"y": 0, "fold": fold, "features": edge["features"]})

    nuisance_check(nuisance_rows, feature_names)
    validate_matching(selected, pos_features, feature_names, fold_by_chrom)
    used37 = sorted({key for bag in bags for key in bag["members"]}, key=lambda k: (variants[k]["chr38"], variants[k]["pos38"], variants[k]["ref"], variants[k]["alt"]))
    rows = []
    key_to_row = {}
    seen38 = {}
    for key37 in used37:
        row = variants[key37]
        key38 = row["key38"]
        if key38 not in seen38:
            variant_row = len(rows)
            seen38[key38] = variant_row
            rows.append({
                "variant_row": variant_row, "variant_key": key38,
                "chr38": row["chr38"], "pos38": row["pos38"],
                "ref": row["ref"], "alt": row["alt"],
                "sequence_available": True, "atac_window_available": True,
            })
        key_to_row[key37] = seen38[key38]

    mapping = []
    for bag in bags:
        ordered = sorted(set(bag["members"]), key=lambda k: (variants[k]["pos38"], variants[k]["ref"], variants[k]["alt"]))
        mapping.extend({"mil_locus_id": bag["mil_locus_id"], "variant_row": key_to_row[k], "bag_order": i} for i, k in enumerate(ordered))

    OUT.mkdir(parents=True, exist_ok=True)
    meta = pl.DataFrame(metadata).sort("fold", "mil_locus_id")
    meta.select("mil_locus_id", "y", "match_set_id", "dependency_group", "group_weight").write_parquet(OUT / "locus_labels.parquet", compression="zstd")
    meta.select("mil_locus_id", "fold").write_csv(OUT / "folds.tsv", separator="\t")
    meta.write_parquet(OUT / "locus_metadata.parquet", compression="zstd")
    pl.DataFrame(rows).write_parquet(OUT / "model_variant_index.parquet", compression="zstd")
    pl.DataFrame(mapping).write_parquet(OUT / "locus_variant_map.parquet", compression="zstd")
    print(f"Stage 1 complete: {len(positives)} positives, {len(selected)} controls (1:{ratio}), {len(rows)} unique variants", flush=True)


def main() -> None:
    required = [
        STAGE0 / "official_signals_b37.parquet", STAGE0 / "official_loci_b37.parquet",
        FASTA, GTF, CHAIN_FWD, CHAIN_REV, BLACKLIST, VEP, VEP_CACHE,
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    signals = pl.read_parquet(STAGE0 / "official_signals_b37.parquet")
    loci = pl.read_parquet(STAGE0 / "official_loci_b37.parquet")
    signals = signals.filter(~((pl.col("chrom") == 6) & pl.col("pos37").is_between(*MHC)))

    work_root = Path(os.environ.get("ASTER_STAGE1_WORK", DATA / "work/stage1"))
    work_root.mkdir(parents=True, exist_ok=True)
    roots = prepare_sumstats(work_root)
    raw_cache = work_root / (
        f"raw_bags_v3_p{CONTROL_CANDIDATES}_l{CONTROL_LD_CANDIDATES}_s{CONTROL_SHORTLIST}.pkl"
    )
    if raw_cache.exists():
        with raw_cache.open("rb") as handle:
            positives, controls, metrics, anchors = pickle.load(handle)
    else:
        positives, controls, metrics, anchors = [], [], {}, {}
        tss_match = parse_genes()
        with ThreadPoolExecutor(max_workers=CHROM_WORKERS) as executor:
            results = executor.map(
                lambda chrom: chromosome_bags(
                    chrom, roots, signals, loci, work_root, tss_match
                ),
                range(1, 23),
            )
            for p, c, m, a in results:
                positives.extend(p); controls.extend(c); metrics.update(m); anchors.update(a)
        with raw_cache.open("wb") as handle:
            pickle.dump((positives, controls, metrics, anchors), handle, protocol=5)
    all_keys = {key for bag in positives for key in bag["members"]} | {key for bag in controls for key in bag["members"]}
    metrics = {key: value for key, value in metrics.items() if key in all_keys}
    variants = liftover_variants(all_keys, metrics, work_root)
    coding = vep_coding(variants, work_root)
    positives, controls = clean_bags(positives, controls, variants, coding)
    selected, pos_features, ratio, feature_names = match_controls(positives, controls, variants, anchors)
    save_outputs(positives, selected, variants, pos_features, ratio, feature_names)
    print(f"Intermediate Stage 1 workspace retained at {work_root}")


if __name__ == "__main__":
    main()
