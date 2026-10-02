# ASTER

ASTER is the final Sequence + cell-type ATAC residual-gated hierarchical MIL model for weakly supervised prioritization of type 2 diabetes GWAS locus bags. It uses frozen 160-dimensional Sei-derived sequence features, two-channel ATAC windows from six non-diabetic (ND6) or six T2D islet multiome donors, and normalized log-sum-exp cell/variant pooling. RNA/Gene is **not** a model input in this version. Scores are priorities, not causal probabilities.

## Install

```bash
conda env create -f environment.yml
conda activate aster
```

## Prepare data

Place the processed ND6 and T2D6 tensors outside Git, for example under `data/ND6/` and `data/T2D6/` (both directories are ignored). Each cohort directory must contain:

```text
stage1/model_variant_index.parquet
stage1/locus_variant_map.parquet
stage1/locus_labels.parquet
stage1/folds.tsv
stage2_multiome/model_contexts.tsv
stage3_sequence/sequence_sei.h5
stage4_atac/atac_features.zarr/
stage6_alignment/cv_contexts.tsv
stage6_alignment/cv_scalers.npz
stage7_dataset/locus_index.parquet
stage7_dataset/labels.parquet
stage7_dataset/bag_variant_rows.npy
stage7_dataset/dataset_manifest.json
```

The two cohorts use the **same T2DGGI 2024 GWAS bags and dependency-safe folds**; their islet multiome references come from GSE200044 and differ in ATAC features and fold-specific ATAC scalers. The sequence file may be shared with `--sequence-file` if it is not duplicated in T2D6. Stage 1/3/4 tensors are external inputs; raw GWAS, FASTQ, Cell Ranger outputs and Sei weights are not stored here. `scripts/build_scalers.py` recreates train-fold-only Stage 6 scalers, and `scripts/prepare_data.py` recreates the Stage 7 compact bag index from those upstream features. They are only needed when these outputs have not already been supplied.

```bash
python scripts/build_scalers.py --data-root data/ND6
python scripts/prepare_data.py --data-root data/ND6
python scripts/check_data.py --data-root data/ND6
```

For T2D6, replace `data/ND6` with `data/T2D6` and pass `--sequence-file data/ND6/stage3_sequence/sequence_sei.h5` to `build_scalers.py`, `check_data.py`, `train.py` and `inference.py` if using a shared sequence file. Newly generated Stage 7 manifests store relative feature names only; the loader resolves features from `--data-root` and never follows local paths embedded in an old manifest. Releasing the external Stage 1/3/4 feature tensors separately is necessary for third parties to rerun training from this repository alone.

## Train and evaluate

Run identical five-fold nested CV separately for each ATAC reference. Dropout is selected from 0.15/0.30/0.45 in the four inner folds; each outer training fold is refit with seeds 2026/2027/2028. Checkpoints and full OOF predictions are written to ignored `outputs/`.

```bash
python scripts/train.py --data-root data/ND6 --output outputs/ND6
python scripts/train.py --data-root data/T2D6 --output outputs/T2D6
python scripts/train.py --model sequence_only --data-root data/ND6 \
  --output outputs/Sequence_only
python scripts/evaluate.py --data-root data/ND6 \
  --predictions outputs/ND6/oof_locus_predictions.parquet \
  --output outputs/ND6/metrics.json
python scripts/evaluate.py --data-root data/T2D6 \
  --predictions outputs/T2D6/oof_locus_predictions.parquet \
  --output outputs/T2D6/metrics.json
python scripts/evaluate.py --data-root data/ND6 \
  --predictions outputs/Sequence_only/oof_locus_predictions.parquet \
  --output outputs/Sequence_only/metrics.json
```

Add `--variant-predictions outputs/ND6/oof_variant_predictions.parquet --benchmark path/to/independent_targets.parquet` to evaluation only when an independent held-out variant benchmark is available. These target labels must never be used for model fitting or dropout selection.

## Inference

Use each outer fold's three refit models only on its own held-out bags:

```bash
python scripts/inference.py --data-root data/ND6 --model-dir outputs/ND6 \
  --outer-fold 0 --output outputs/ND6/inference_fold0
```

The command writes bag probabilities and within-bag variant scores. Test labels are intentionally inaccessible to the Dataset; `evaluate.py` joins them after OOF inference.

## Repository

`src/` contains the Dataset, final model, loss and training engine; `scripts/` contains Stage 6/7 preparation and CLI entry points; `configs/paper.yml` freezes paper hyperparameters; `examples/paper_metrics.tsv` holds a small reference summary. No raw data, checkpoints, training curves or historical experiments are tracked.

## Citation

*Cell-Contextual Hierarchical Multiple-Instance Learning for Type 2 Diabetes GWAS Locus Prioritization* (manuscript in preparation). Please cite the published paper when bibliographic details become available.
