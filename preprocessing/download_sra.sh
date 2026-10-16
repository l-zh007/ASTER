#!/usr/bin/env bash
set -euo pipefail

root="${1:?Usage: download_sra.sh COHORT_MULTIOME_DIR}"
runs="$root/00_metadata/runs.tsv"
destination="$root/02_sra"
mkdir -p "$destination"
command -v prefetch >/dev/null
command -v vdb-validate >/dev/null
while IFS=$'\t' read -r donor modality lane run extra; do
    [[ "$donor" == "donor_id" || -z "$run" ]] && continue
    file="$destination/$run/$run.sra"
    if [[ ! -s "$file" ]]; then
        prefetch "$run" --output-directory "$destination/$run"
    fi
    vdb-validate "$file"
    printf '%s\t%s\t%s\tcomplete\n' "$donor" "$modality" "$run"
done < "$runs"
