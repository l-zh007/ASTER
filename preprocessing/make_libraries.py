#!/usr/bin/env python3
"""Write Cell Ranger ARC library CSVs from the fixed public donor manifests."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("ND6", "T2D6"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    root = args.data_root.resolve() / args.cohort / "multiome"
    manifest = root / "00_metadata/samples.tsv"
    target = root / "04_cellranger_arc/config"
    target.mkdir(parents=True, exist_ok=True)
    with manifest.open() as handle:
        donors = list(csv.DictReader(handle, delimiter="\t"))
    if len(donors) != 6:
        raise ValueError("Expected six donors in the cohort manifest")
    for donor in donors:
        name = donor["donor_id"]
        output = target / f"{name}.libraries.csv"
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fastqs", "sample", "library_type"])
            for modality, kind in (("ATAC", "Chromatin Accessibility"), ("GEX", "Gene Expression")):
                fastqs = (root / "03_fastq" / name / modality).resolve()
                writer.writerow([fastqs, f"{name}_{modality}", kind])
        print(output)


if __name__ == "__main__":
    main()
