#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?Usage: prepare_fastq.sh COHORT_MULTIOME_DIR}"
RUNS="$ROOT/00_metadata/runs.tsv"
SRA_DIR="$ROOT/02_sra"
FASTQ_DIR="$ROOT/03_fastq"
TMP_ROOT="$ROOT/tmp/fasterq"
SRA_BIN="${SRA_BIN:-$(dirname "$(command -v fasterq-dump)")}"
THREADS="${THREADS:-24}"
PIGZ_THREADS="${PIGZ_THREADS:-24}"
FASTQ_JOBS="${FASTQ_JOBS:-1}"

export PATH="$SRA_BIN:$PATH"
mkdir -p "$FASTQ_DIR" "$TMP_ROOT"

convert_one() {
    local donor="$1"
    local modality="$2"
    local lane="$3"
    local run="$4"
    local sra outdir sample lane_name expected ready read_no target work count
    local r1_len r2_len r3_len source_no

    sra="$SRA_DIR/$run/$run.sra"
    outdir="$FASTQ_DIR/$donor/$modality"
    sample="${donor}_${modality}"
    lane_name=$(printf 'L%03d' "$lane")
    expected=2
    [[ "$modality" == "ATAC" ]] && expected=3
    mkdir -p "$outdir"

    ready=1
    for read_no in $(seq 1 "$expected"); do
        target="$outdir/${sample}_S1_${lane_name}_R${read_no}_001.fastq.gz"
        [[ -s "$target" ]] && gzip -t "$target" 2>/dev/null || ready=0
    done
    if [[ "$ready" -eq 1 ]]; then
        printf '%s\t%s\tFASTQ already complete\n' "$donor" "$run"
        return 0
    fi

    [[ -s "$sra" ]] || { printf 'Missing SRA: %s\n' "$sra" >&2; exit 1; }
    vdb-validate "$sra" >/dev/null

    work=$(find "$TMP_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${run}.*" \
        -exec test -s '{}/out/'"${run}"'_1.fastq' ';' -print -quit)
    if [[ -z "$work" ]]; then
        work=$(mktemp -d "$TMP_ROOT/${run}.XXXXXX")
        mkdir -p "$work/out" "$work/tmp"
        printf '%s\t%s\tconversion started %s\n' "$donor" "$run" "$(date -Is)"
        fasterq-dump "$sra" \
            --include-technical \
            --split-files \
            --threads "$THREADS" \
            --temp "$work/tmp" \
            --outdir "$work/out" \
            --size-check on
    else
        printf '%s\t%s\treusing completed temporary FASTQ extraction\n' "$donor" "$run"
    fi

    count=$(find "$work/out" -maxdepth 1 -type f -name "${run}_*.fastq" | wc -l)
    r1_len=$(awk 'NR==2 {print length($0); exit}' "$work/out/${run}_1.fastq")
    if [[ "$modality" == "ATAC" ]]; then
        [[ "$count" -eq 3 ]] || {
            printf '%s produced %d ATAC reads; expected 3\n' "$run" "$count" >&2
            return 1
        }
        r2_len=$(awk 'NR==2 {print length($0); exit}' "$work/out/${run}_2.fastq")
        r3_len=$(awk 'NR==2 {print length($0); exit}' "$work/out/${run}_3.fastq")
        [[ "$r1_len" -ge 28 && ( "$r2_len" -eq 10 || "$r2_len" -eq 24 ) && "$r3_len" -ge 28 ]] || {
            printf '%s invalid ATAC read structure: %s/%s/%s\n' "$run" "$r1_len" "$r2_len" "$r3_len" >&2
            return 1
        }
    else
        [[ "$count" -eq 2 || "$count" -eq 3 ]] || {
            printf '%s produced %d GEX reads; expected 2 biological reads with an optional technical read\n' "$run" "$count" >&2
            return 1
        }
        source_no=2
        [[ "$count" -eq 3 ]] && source_no=3
        r2_len=$(awk 'NR==2 {print length($0); exit}' "$work/out/${run}_${source_no}.fastq")
        [[ "$r1_len" -ge 28 && "$r2_len" -ge 50 ]] || {
            printf '%s invalid GEX read structure: %s/%s\n' "$run" "$r1_len" "$r2_len" >&2
            return 1
        }
    fi

    for read_no in $(seq 1 "$expected"); do
        source_no="$read_no"
        [[ "$modality" == "GEX" && "$count" -eq 3 && "$read_no" -eq 2 ]] && source_no=3
        target="$outdir/${sample}_S1_${lane_name}_R${read_no}_001.fastq"
        mv "$work/out/${run}_${source_no}.fastq" "$target"
        pigz -p "$PIGZ_THREADS" -f "$target"
        gzip -t "$target.gz"
    done
    find "$work" -depth -type d -empty -delete
    printf '%s\t%s\tcomplete\tread_lengths=%s/%s' "$donor" "$run" "$r1_len" "$r2_len"
    [[ "$modality" == "ATAC" ]] && printf '/%s' "$r3_len"
    printf '\n'
}

export ROOT RUNS SRA_DIR FASTQ_DIR TMP_ROOT SRA_BIN THREADS PIGZ_THREADS
export -f convert_one
awk -F '\t' 'NR > 1 {print $1, $2, $3, $4}' OFS='\t' "$RUNS" \
    | xargs -P "$FASTQ_JOBS" -n 4 bash -c 'convert_one "$1" "$2" "$3" "$4"' _

printf 'All cohort FASTQs completed: %s\n' "$(date -Is)"
