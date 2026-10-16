# ASTER

Code and compact source results for *ASTER Integrates Sequence and Islet Accessibility for Type 2 Diabetes GWAS Signals*.

ASTER learns from **T2D GWAS signal bags**, not labeled causal variants. Frozen Sei supplies 160 sequence features; donor-balanced islet ATAC supplies eight cell-context profiles of 100 bins × two channels (accessibility and **cell-support fraction**). An ATAC residual gate and direct fusion yield variant–cell scores, followed by masked normalized log-sum-exp pooling over cells and variants to predict each bag. The paper compares the same GWAS bags under six non-diabetic donors (ND6), six T2D donors (T2D6), and a Sequence-only control. RNA is used to inherit the published multiome cell annotations; it is **not** a model branch. Variant and cell scores are relative priorities, not causal probabilities.

## Install

Linux with conda and a CUDA-capable GPU is recommended for the full pipeline. From this directory:

```bash
conda env create -f environment.yml
conda activate aster
python scripts/evaluate.py --published-results results
```

For frozen Sei feature extraction, run `conda env create -f environment-sei.yml` and activate `aster-sei` for Stage 3. The raw-data route also requires `bcftools`, PLINK 2, UCSC `liftOver`, Ensembl VEP with a GRCh38 release-116 offline cache, SRA Toolkit, `pigz`, `g++`, and [Cell Ranger ARC 2.2.0](https://www.10xgenomics.com/support/software/cell-ranger-arc/downloads). These third-party programs, references and pretrained weights are not redistributed here.

## Data preparation

Download the following from their providers; keep all raw and intermediate files in ignored `data/`:

| Input | Source | Local placement |
| --- | --- | --- |
| 2024 T2DGGI all-ancestry and AFA/EAS/EUR/HIS/SAS GWAS summary statistics; Suzuki supplementary ST4/ST6 workbook | [DIAGRAM/T2DGGI](https://www.diagram-consortium.org/downloads.html), [Suzuki et al.](https://doi.org/10.1038/s41586-024-07019-6) | `data/gwas/`; workbook in `data/gwas/supplementary/`. Preserve the six filenames expected by `preprocessing/stage0.py`. |
| GSE200044 paired ATAC/GEX SRA runs for twelve selected donors; authors' `GSE200044_cell_cluster.tar.gz` | [GEO GSE200044](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200044), [official cluster archive](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE200nnn/GSE200044/suppl/GSE200044_cell_cluster.tar.gz); exact runs in `preprocessing/manifests/` | Cohort SRA/FASTQ under `data/{ND6,T2D6}/multiome/`; extract `cell_cluster/multiome_RNA.h5ad` to `data/reference/GSE200044/`. |
| 1000 Genomes Phase 3 GRCh37 VCFs and sample panel; 10x ARC GRCh38 reference; hg19↔hg38 chains; hg38 blacklist; VEP cache | [1000 Genomes](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/), [10x ARC reference](https://www.10xgenomics.com/support/software/cell-ranger-arc/downloads), [UCSC forward chain](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz) and [reverse chain](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz), [ENCODE GRCh38 blacklist](https://www.encodeproject.org/files/ENCFF356LFX/), [Ensembl VEP](https://www.ensembl.org/info/docs/tools/vep/index.html) | `data/reference/` as resolved in `preprocessing/stage0.py` and `stage1.py`; use the **same GRCh38 ARC reference** for Cell Ranger and variant features. |
| Frozen Sei code and weights | [Sei framework](https://github.com/FunctionLab/sei-framework), commit `1e16bff9b1618af7065a23f82c4a3bb82aaeb689` | Set `ASTER_SEI_DIR`; run its upstream weight download. No Sei training is performed here. |

The full raw-data route is compute- and storage-intensive, principally because GSE200044 fragments are regenerated from ATAC/GEX SRA runs. A prepared-input route is also possible: supply the shared Stage 1/3 files and each cohort's official-cell/ATAC files with the layout below, then run only Stage 6/7. Prepared large tensors are **not** hosted in this repository.

```text
data/
  shared/stage1/{locus_labels.parquet,locus_variant_map.parquet,model_variant_index.parquet,folds.tsv}
  shared/stage3_sequence/sequence_sei.h5
  ND6/{stage2_multiome/model_contexts.tsv,stage4_atac/atac_features.zarr}
  T2D6/{stage2_multiome/model_contexts.tsv,stage4_atac/atac_features.zarr}
```

To regenerate the inputs, run from the repository root (the manifests fix the donor/run selection):

```bash
export ASTER_DATA_ROOT="$PWD/data"
export ASTER_SEI_DIR=/path/to/sei-framework
export ASTER_VEP_CACHE="$ASTER_DATA_ROOT/reference/vep_cache"
export ASTER_BLACKLIST="$ASTER_DATA_ROOT/reference/hg38.blacklist.bed.gz"
export ARC_REFERENCE="$ASTER_DATA_ROOT/reference/GRCh38_10x_ARC"
for cohort in ND6 T2D6; do
  mkdir -p "data/$cohort/multiome/00_metadata"
  cp "preprocessing/manifests/${cohort}_samples.tsv" "data/$cohort/multiome/00_metadata/samples.tsv"
  cp "preprocessing/manifests/${cohort}_runs.tsv" "data/$cohort/multiome/00_metadata/runs.tsv"
done
python preprocessing/stage0.py
python preprocessing/stage1.py
conda activate aster-sei
python preprocessing/stage3.py --mode full
conda activate aster
for cohort in ND6 T2D6; do
  bash preprocessing/download_sra.sh "data/$cohort/multiome"
  bash preprocessing/prepare_fastq.sh "data/$cohort/multiome"
  python preprocessing/make_libraries.py --cohort "$cohort"
  for donor in $(tail -n +2 "preprocessing/manifests/${cohort}_samples.tsv" | cut -f1); do
    bash preprocessing/run_cellranger_one.sh "data/$cohort/multiome" "$donor"
  done
  ASTER_COHORT="$cohort" python preprocessing/stage2.py
  ASTER_COHORT="$cohort" python preprocessing/stage4.py
  ln -s ../shared/stage1 "data/$cohort/stage1"
  ln -s ../shared/stage3_sequence "data/$cohort/stage3_sequence"
  python scripts/build_scalers.py --data-root "data/$cohort"
  python scripts/prepare_data.py --data-root "data/$cohort"
  python scripts/check_data.py --data-root "data/$cohort"
done
```

Set `ASTER_VEP` if `vep` is not on `PATH`; `ASTER_BCFTOOLS`, `ASTER_PLINK` and `ASTER_LIFTOVER` can likewise override tool locations. Stage 1 performs GRCh37→GRCh38 allele checks, noncoding signal/control bag construction, 1:2 matching and dependency-safe folds. Stage 2 inherits the official eight broad cell classes without reclustering. Stage 3 freezes Sei. Stage 4 builds donor-equal ATAC profiles. Stage 6 fits scalers **only on training-fold unique variants**; Stage 7 builds the PyTorch bag index. There is no Gene/RNA Stage 5 in the final model. For a prepared-input route, link shared Stage 1/3 into each cohort, then run the final three `scripts/` commands only.

## Train and evaluate

`configs/paper.yml` fixes the five outer folds, four inner folds, dropout choices, three refit seeds, optimizer and loss settings. The same bags, dependency groups, masks and folds are used for all three models. Each command writes ignored checkpoints and OOF predictions under `outputs/`.

```bash
python scripts/train.py --data-root data/ND6 --output outputs/ND6
python scripts/train.py --data-root data/T2D6 --output outputs/T2D6
python scripts/train.py --model sequence_only --data-root data/ND6 --output outputs/Sequence_only
python scripts/evaluate.py --data-root data/ND6 --predictions outputs/ND6/oof_locus_predictions.parquet --output outputs/ND6/metrics.json
python scripts/evaluate.py --data-root data/T2D6 --predictions outputs/T2D6/oof_locus_predictions.parquet --output outputs/T2D6/metrics.json
python scripts/evaluate.py --data-root data/ND6 --predictions outputs/Sequence_only/oof_locus_predictions.parquet --output outputs/Sequence_only/metrics.json
```

For the manuscript's post hoc 137-bag DIAMANTE comparison, add `--variant-predictions outputs/<model>/oof_variant_predictions.parquet --benchmark results/diamante_development_targets.tsv.gz` to an evaluation command. This partly overlapping statistical target is **not** an independent functional validation set and was not a training label. To score held-out fold 0 again and export its top cell context, for example:

```bash
python scripts/inference.py --data-root data/ND6 --model-dir outputs/ND6 --outer-fold 0 --cell-contexts --output outputs/ND6/outer_0/context
```

Repeat for folds 1–4 and T2D6, then export the compact result tables from new OOF runs:

```bash
python scripts/export_results.py --data-root data/ND6 \
  --sequence-only outputs/Sequence_only --nd6 outputs/ND6 --t2d6 outputs/T2D6 \
  --benchmark results/diamante_development_targets.tsv.gz --output outputs/paper_tables
```

The published tables can also be checked immediately, without downloading large tensors:

```bash
python scripts/evaluate.py --published-results results
python scripts/plot_results.py --results results --output outputs/fig2.png
```

This recomputes bag AUROC/AUPRC and matched top-1, DIAMANTE MRR/Recall@k, ND6–T2D6 rank concordance and endocrine top-context fractions from the included OOF results. The expected bag AUROC/AUPRC values are 0.6611/0.4707 (Sequence-only), 0.7043/0.5451 (ND6) and 0.7044/0.5413 (T2D6). **These are bag-discrimination metrics, not variant-level accuracy.**

## Files and availability

`preprocessing/` contains only final data-generation scripts and exact donor/run lists; `src/` contains Dataset, models, loss and training engine; `scripts/` contains fold scaling, bag indexing, nested CV, inference and evaluation; `configs/` fixes the paper run. `results/` contains compact OOF bag predictions, per-bag variant scores/ranks and top-cell assignments, DIAMANTE statistical targets for the development comparison, case rows and summary metrics. It contains no raw multiome reads, fragments, large feature tensors, training logs or checkpoints.

**Data availability.** Original T2DGGI GWAS and DIAMANTE fine-mapping data are available from [DIAGRAM/T2DGGI](https://www.diagram-consortium.org/downloads.html); original paired multiome data and annotation are available under [GSE200044](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200044). The small derived source results listed above are in `results/`. Large derived training tensors are not redistributed; the required sources, placement and preprocessing code are specified here.

**Code availability.** The final method and commands are in this [ASTER repository](https://github.com/l-zh007/ASTER). Cite a versioned release/DOI when one is deposited; no DOI is claimed yet.

**Citation.** Please cite *ASTER Integrates Sequence and Islet Accessibility for Type 2 Diabetes GWAS Signals* when published. Also cite the original [T2DGGI](https://doi.org/10.1038/s41586-024-07019-6), [GSE200044](https://doi.org/10.1038/s41588-023-01397-9) and [Sei](https://doi.org/10.1038/s41588-022-01102-2) studies when reusing their data or model.
