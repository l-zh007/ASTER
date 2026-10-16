#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?Usage: run_cellranger_one.sh COHORT_MULTIOME_DIR SAMP-XXXX}"
donor="${2:?Usage: run_cellranger_one.sh COHORT_MULTIOME_DIR SAMP-XXXX}"
CELLRANGER="${CELLRANGER_ARC:-cellranger-arc}"
REFERENCE="${ARC_REFERENCE:?Set ARC_REFERENCE to refdata-cellranger-arc-GRCh38-2020-A-2.0.0}"
LIBRARIES="$ROOT/04_cellranger_arc/config/${donor}.libraries.csv"
OUTPUT_ROOT="$ROOT/04_cellranger_arc"
CORES="${CELLRANGER_CORES:-48}"
MEM_GB="${CELLRANGER_MEM_GB:-300}"

command -v "$CELLRANGER" >/dev/null || { printf 'Cell Ranger ARC is not installed: %s\n' "$CELLRANGER" >&2; exit 1; }
[[ -f "$REFERENCE/reference.json" ]] || { printf 'Invalid reference: %s\n' "$REFERENCE" >&2; exit 1; }
[[ -s "$LIBRARIES" ]] || { printf 'Missing libraries CSV: %s\n' "$LIBRARIES" >&2; exit 1; }

for modality in ATAC GEX; do
    find "$ROOT/03_fastq/$donor/$modality" -maxdepth 1 -type f -name '*.fastq.gz' -print -quit | grep -q . || {
        printf 'Missing %s FASTQs for %s\n' "$modality" "$donor" >&2
        exit 1
    }
done

if [[ -s "$OUTPUT_ROOT/$donor/outs/atac_fragments.tsv.gz" \
      && -s "$OUTPUT_ROOT/$donor/outs/atac_fragments.tsv.gz.tbi" \
      && -s "$OUTPUT_ROOT/$donor/outs/filtered_feature_bc_matrix.h5" ]]; then
    printf '%s Cell Ranger outputs already complete\n' "$donor"
    exit 0
fi

cd "$OUTPUT_ROOT"
"$CELLRANGER" count \
    --id "$donor" \
    --description "GSE200044 donor $donor; GRCh38 2020-A" \
    --reference "$REFERENCE" \
    --libraries "$LIBRARIES" \
    --localcores "$CORES" \
    --localmem "$MEM_GB" \
    --create-bam false \
    --disable-ui

gzip -t "$OUTPUT_ROOT/$donor/outs/atac_fragments.tsv.gz"
[[ -s "$OUTPUT_ROOT/$donor/outs/atac_fragments.tsv.gz.tbi" ]]
[[ -s "$OUTPUT_ROOT/$donor/outs/filtered_feature_bc_matrix.h5" ]]
printf '%s Cell Ranger ARC complete: %s\n' "$donor" "$(date -Is)"
