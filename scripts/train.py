#!/usr/bin/env python3
"""Five-fold nested CV for ND6 or T2D6 ASTER features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CachedDataset, LocusDataset  # noqa: E402
from src.engine import fit_inner, predict, refit, seed_all  # noqa: E402
from src.model import ASTER, SequenceOnly  # noqa: E402


def load_data(root, context, partition, sequence_file):
    return CachedDataset(LocusDataset(root, context, partition, sequence_file))


def signature(config_file, data_root, model_kind):
    repo = Path(__file__).resolve().parents[1]
    files = [config_file, repo / "src/model.py", repo / "src/data.py",
             repo / "src/engine.py", repo / "src/losses.py", Path(__file__),
             data_root / "stage6_alignment/alignment_manifest.json",
             data_root / "stage7_dataset/dataset_manifest.json"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    digest.update(model_kind.encode())
    return digest.hexdigest()


def save_checkpoint(path, model, model_config, model_kind, dropout, seed, context, epochs, sig, history):
    payload = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "model_config": model_config, "model_kind": model_kind,
               "dropout": dropout, "seed": seed,
               "cv_context_id": context, "refit_epoch": epochs,
               "run_signature": sig, "training_history": history}
    torch.save(payload, path)


def train_fold(fold, root, sequence_file, out, config, device, sig, model_kind):
    model_config, settings = config["model"], config["training"]
    model_class = ASTER if model_kind == "aster" else SequenceOnly
    candidates = (settings["dropout_candidates"] if model_kind == "aster" else [0.15])
    fold_dir = out / f"outer_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    inner_path = fold_dir / "inner_selection.tsv"
    inner = pd.read_csv(inner_path, sep="\t") if inner_path.exists() else pd.DataFrame()
    if not inner.empty and set(inner.run_signature) != {sig}:
        raise ValueError("Existing inner selection belongs to different code/config")

    for inner_fold in settings["outer_folds"]:
        if inner_fold == fold:
            continue
        context = f"outer_{fold}__inner_{inner_fold}"
        train = validation = None
        for dropout in candidates:
            if not inner.empty and ((inner.cv_context_id == context) &
                                    (inner.dropout == dropout)).any():
                continue
            if train is None:
                train = load_data(root, context, "train", sequence_file)
                validation = load_data(root, context, "validation", sequence_file)
            seed_all(2026)
            model = model_class(model_config, dropout).to(device)
            result = fit_inner(model, train, validation, device, settings)
            result.update(cv_context_id=context, dropout=dropout, run_signature=sig)
            inner = pd.concat([inner, pd.DataFrame([result])], ignore_index=True)
            inner.sort_values(["cv_context_id", "dropout"]).to_csv(inner_path, sep="\t", index=False)
        del train, validation

    summary = (inner.groupby("dropout", as_index=False)
               .agg(mean_validation_loss=("validation_loss_at_best", "mean"),
                    median_best_epoch=("best_epoch", "median"))
               .sort_values(["mean_validation_loss", "dropout"]))
    if len(summary) != len(candidates):
        raise ValueError("Incomplete inner CV")
    dropout = float(summary.iloc[0].dropout)
    selected = inner.loc[inner.dropout.eq(dropout)]
    epochs = max(1, math.floor(float(selected.best_epoch.median()) + 0.5))
    (fold_dir / "selected_config.json").write_text(json.dumps({
        "outer_fold": fold, "model_kind": model_kind,
        "dropout": dropout, "refit_epoch": epochs,
        "run_signature": sig,
    }, indent=2) + "\n")

    context = f"outer_{fold}__refit"
    train = load_data(root, context, "train", sequence_file)
    test = load_data(root, context, "test", sequence_file)
    models = []
    for seed in settings["refit_seeds"]:
        path = fold_dir / f"seed_{seed}.pt"
        seed_all(seed)
        model = model_class(model_config, dropout).to(device)
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (saved["run_signature"], saved["cv_context_id"], saved["refit_epoch"],
                saved["dropout"], saved["seed"]) != (sig, context, epochs, dropout, seed):
                raise ValueError(f"Incompatible checkpoint: {path}")
            model.load_state_dict(saved["model_state"])
        else:
            history = refit(model, train, device, settings, seed, epochs)
            save_checkpoint(path, model, model_config, model_kind,
                            dropout, seed, context, epochs, sig, history)
        models.append(model)
    locus, variant = predict(models, test, device, fold, settings["refit_seeds"])
    locus.to_parquet(fold_dir / "test_locus_predictions.parquet", index=False)
    variant.to_parquet(fold_dir / "test_variant_predictions.parquet", index=False)
    del train, test, models
    if device.type == "cuda":
        torch.cuda.empty_cache()


def combine(out, folds):
    for kind in ("locus", "variant"):
        frames = [pd.read_parquet(out / f"outer_{fold}/test_{kind}_predictions.parquet")
                  for fold in folds]
        combined = pd.concat(frames, ignore_index=True)
        keys = ["mil_locus_id"] + (["variant_row"] if kind == "variant" else [])
        if combined.duplicated(keys).any():
            raise ValueError(f"Duplicate OOF {kind} predictions")
        combined.sort_values(keys).to_parquet(out / f"oof_{kind}_predictions.parquet", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing stage1/, stage3_sequence/, stage4_atac/, stage6_alignment/, stage7_dataset/")
    parser.add_argument("--sequence-file", type=Path, help="Optional shared sequence_sei.h5 for T2D6")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/paper.yml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", choices=("aster", "sequence_only"), default="aster")
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    folds = tuple(config["training"]["outer_folds"])
    if args.combine_only:
        combine(args.output, folds)
        return
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    sig = signature(args.config, args.data_root, args.model)
    selected_folds = tuple(args.outer_fold) if args.outer_fold else folds
    for fold in selected_folds:
        train_fold(fold, args.data_root, args.sequence_file,
                   args.output, config, device, sig, args.model)
    if set(selected_folds) == set(folds):
        combine(args.output, folds)


if __name__ == "__main__":
    main()
