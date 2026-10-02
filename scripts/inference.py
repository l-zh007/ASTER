#!/usr/bin/env python3
"""Re-score one held-out outer fold using its three refit checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CachedDataset, LocusDataset  # noqa: E402
from src.engine import predict  # noqa: E402
from src.model import ASTER, SequenceOnly  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True, choices=range(5))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(8)
    fold_dir = args.model_dir / f"outer_{args.outer_fold}"
    selected = json.loads((fold_dir / "selected_config.json").read_text())
    seeds = (2026, 2027, 2028)
    models = []
    for seed in seeds:
        checkpoint = torch.load(fold_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
        if checkpoint["cv_context_id"] != f"outer_{args.outer_fold}__refit":
            raise ValueError("Checkpoint and requested fold disagree")
        model_config = checkpoint.get("model_config", checkpoint.get("architecture_config"))
        dropout = checkpoint.get("dropout", selected["dropout"])
        if dropout != selected["dropout"] or checkpoint["refit_epoch"] != selected["refit_epoch"]:
            raise ValueError("Checkpoint does not match the outer-fold selected model")
        model_kind = checkpoint.get("model_kind", selected.get("model_kind", "aster"))
        model_class = ASTER if model_kind == "aster" else SequenceOnly
        model = model_class(model_config, dropout).to(device)
        model.load_state_dict(checkpoint["model_state"])
        models.append(model)
    data = CachedDataset(LocusDataset(args.data_root, f"outer_{args.outer_fold}__refit",
                                      "test", args.sequence_file))
    locus, variant = predict(models, data, device, args.outer_fold, seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    locus.to_parquet(args.output / "locus_predictions.parquet", index=False)
    variant.to_parquet(args.output / "variant_predictions.parquet", index=False)
    print(f"Scored {len(locus)} held-out bags and {len(variant)} variant memberships")


if __name__ == "__main__":
    main()
