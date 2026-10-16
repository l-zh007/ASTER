#!/usr/bin/env python3
"""T2DGGI 2024 Stage 0: freeze official signals/loci and LD sample sets."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ASTER_DATA_ROOT", ROOT / "data")).resolve()
GWAS = DATA / "gwas"
MULTI = GWAS / "Suzuki.Nature2024.T2DGGI.MultiAncestry.sumstats.zip"
SUPP = GWAS / "supplementary/41586_2024_7019_MOESM3_ESM.xlsx"
REF = DATA / "reference"
LD = REF / "1000G_phase3_GRCh37"
OUT = DATA / "shared/stage0"
BCFTOOLS = os.environ.get("ASTER_BCFTOOLS", "bcftools")

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
BASES = {"A", "C", "G", "T"}


def require(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing required input:\n" + "\n".join(missing))


def xlsx_rows(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read one XLSX sheet with stdlib only; preserve merged-cell blanks."""
    with ZipFile(path) as zf:
        shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in si.iter(NS + "t"))
            for si in shared_root.findall(NS + "si")
        ]
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        targets = {node.attrib["Id"]: node.attrib["Target"] for node in rels}
        sheet = next(
            node for node in workbook.find(NS + "sheets")
            if node.attrib["name"] == sheet_name
        )
        target = targets[sheet.attrib[REL_NS + "id"]]
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        root = ET.fromstring(zf.read(target))
        rows: list[dict[str, str]] = []
        for row in root.find(NS + "sheetData"):
            values: dict[str, str] = {"_row": row.attrib["r"]}
            for cell in row.findall(NS + "c"):
                col = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                value_node = cell.find(NS + "v")
                value = "" if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(x.text or "" for x in cell.iter(NS + "t"))
                values[col] = value.strip() if isinstance(value, str) else value
            rows.append(values)
    return rows


def parse_interval(value: str) -> tuple[int, int]:
    left, right = value.replace(",", "").split("-")
    return int(left), int(right)


def parse_official_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    st4 = xlsx_rows(SUPP, "ST4")
    st6 = xlsx_rows(SUPP, "ST6")
    clusters = {
        (row.get("C", ""), int(float(row["D"]))): row.get("E", "")
        for row in st6
        if int(row["_row"]) >= 5 and row.get("C") and row.get("D") and row.get("E")
    }

    carry = {key: "" for key in "ABCD"}
    locus_number = 0
    signals: list[dict] = []
    loci: list[dict] = []
    for row in st4:
        if int(row["_row"]) < 5:
            continue
        if not row.get("E") or not row.get("F") or not row.get("J"):
            continue
        if row.get("A"):
            locus_number += 1
            for key in carry:
                carry[key] = row.get(key, "")
            start, end = parse_interval(carry["C"])
            loci.append({
                "official_locus_id": f"T2DGGI_LOCUS_{locus_number:04d}",
                "locus_name": carry["A"].strip(),
                "chrom": int(float(carry["B"])),
                "start37": start,
                "end37": end,
                "previously_reported": carry["D"],
            })
        if locus_number == 0:
            continue
        signal_number = len(signals) + 1
        risk, other = row["G"].upper(), row["H"].upper()
        signals.append({
            "official_signal_id": f"T2DGGI_SIGNAL_{signal_number:04d}",
            "official_locus_id": f"T2DGGI_LOCUS_{locus_number:04d}",
            "index_rsid": row["E"],
            "chrom": int(float(carry["B"])),
            "pos37": int(float(row["F"])),
            "risk_allele": risk,
            "other_allele": other,
            "risk_allele_frequency": float(row["I"]),
            "p_assoc": float(row["J"]),
            "fixed_effect_or_ci": row["K"],
            "neff": float(row["L"]),
            "mechanistic_cluster": clusters.get((row["E"], int(float(row["F"]))), ""),
        })

    signal_df, locus_df = pl.DataFrame(signals), pl.DataFrame(loci)
    assert signal_df.height == 1289, signal_df.height
    assert locus_df.height == 611, locus_df.height
    assert signal_df["official_signal_id"].n_unique() == 1289
    assert locus_df["official_locus_id"].n_unique() == 611
    assert signal_df.filter(pl.col("p_assoc") >= 5e-8).is_empty()
    return signal_df, locus_df


def verify_official_signals(signals: pl.DataFrame) -> None:
    with ZipFile(MULTI) as zf, tempfile.TemporaryDirectory() as tmp:
        member = zf.namelist()[0]
        text_path = Path(tmp) / Path(member).name
        with zf.open(member) as source, text_path.open("wb") as target:
            while block := source.read(16 << 20):
                target.write(block)
        wanted = signals.select(pl.col("chrom").alias("Chromsome"), pl.col("pos37").alias("Position"))
        observed = (
            pl.scan_csv(text_path, separator="\t", infer_schema_length=1000)
            .select("Chromsome", "Position", "EffectAllele", "NonEffectAllele", "PvalAssociation")
            .join(wanted.lazy(), on=["Chromsome", "Position"], how="inner")
            .collect(engine="streaming")
            .with_columns(
                pl.col("EffectAllele").str.to_uppercase(),
                pl.col("NonEffectAllele").str.to_uppercase(),
            )
        )
    official = signals.select(
        pl.col("chrom").alias("Chromsome"),
        pl.col("pos37").alias("Position"),
        "risk_allele", "other_allele",
    )
    joined = official.join(observed, on=["Chromsome", "Position"], how="left")
    ok = joined.filter(
        pl.col("EffectAllele").is_not_null()
        & (pl.col("PvalAssociation") < 5e-8)
        & (
            (
                (pl.col("risk_allele") == pl.col("EffectAllele"))
                & (pl.col("other_allele") == pl.col("NonEffectAllele"))
            )
            | (
                (pl.col("risk_allele") == pl.col("NonEffectAllele"))
                & (pl.col("other_allele") == pl.col("EffectAllele"))
            )
        )
    )
    if ok.height != 1289:
        raise RuntimeError(f"Only {ok.height}/1289 official signals reconcile with multi-ancestry GWAS")


def write_ld_keep_files() -> None:
    panel_path = LD / "integrated_call_samples_v3.20130502.ALL.panel"
    related_path = LD / "20140625_related_individuals.txt"
    with related_path.open() as handle:
        related = {
            row["Sample"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("Sample")
        }
    samples: dict[str, list[str]] = {pop: [] for pop in ("AFR", "AMR", "EAS", "EUR", "SAS")}
    with panel_path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sample, pop = row["sample"], row["super_pop"]
            if pop in samples and sample not in related:
                samples[pop].append(sample)

    vcf = LD / "vcf/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    observed = set(subprocess.check_output([BCFTOOLS, "query", "-l", vcf], text=True).splitlines())
    keep_dir = OUT / "ld_samples"
    keep_dir.mkdir(parents=True, exist_ok=True)
    for pop, ids in samples.items():
        if not ids or not set(ids) <= observed:
            raise RuntimeError(f"Invalid {pop} sample set")
        (keep_dir / f"{pop}.keep").write_text("".join(f"{x}\t{x}\n" for x in ids))


def verify_required_inputs() -> None:
    gwas_files = [
        MULTI,
        GWAS / "Suzuki.Nature2024.T2DGGI.AFA.sumstats.zip",
        GWAS / "Suzuki.Nature2024.T2DGGI.EAS.sumstats.zip",
        GWAS / "EUR_Metal_LDSC-CORR_Neff.v2.txt",
        GWAS / "HIS_Metal_LDSC-CORR_Neff.v2.txt",
        GWAS / "SAS_Metal_LDSC-CORR_Neff.v2.txt",
        SUPP,
    ]
    refs = [
        REF / "GRCh38_10x_ARC/fasta/genome.fa",
        REF / "GRCh38_10x_ARC/fasta/genome.fa.fai",
        REF / "GRCh38_10x_ARC/genes/genes.gtf.gz",
        REF / "hg19ToHg38.over.chain.gz",
        LD / "integrated_call_samples_v3.20130502.ALL.panel",
        LD / "20140625_related_individuals.txt",
    ]
    refs += [
        LD / f"vcf/ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz{suffix}"
        for chrom in range(1, 23)
        for suffix in ("", ".tbi")
    ]
    require(gwas_files + refs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    verify_required_inputs()
    signals, loci = parse_official_tables()
    verify_official_signals(signals)
    write_ld_keep_files()
    signals.write_parquet(OUT / "official_signals_b37.parquet", compression="zstd")
    loci.write_parquet(OUT / "official_loci_b37.parquet", compression="zstd")
    print(f"Stage 0 complete: {signals.height} signals, {loci.height} publication loci")


if __name__ == "__main__":
    main()
