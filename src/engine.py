"""Dependency-group optimizer steps and fold-blind ensemble prediction."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from .data import collate, dependency_groups, group_denominator, match_batches
from .losses import numerator


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def model_inputs(model, batch, device):
    return {key: batch[key].to(device) for key in model.input_keys}


def run_epoch(model, dataset, optimizer, device, settings, *, seed=2026, epoch=0, scaler=None):
    training = optimizer is not None
    model.train(training)
    amp = training and device.type == "cuda"
    if training and scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=4096.0)
    losses = []
    groups = dependency_groups(dataset, shuffle=training, seed=seed, epoch=epoch)
    for group in groups:
        denom = group_denominator(dataset, group, settings["positive_weight"]).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for batch in match_batches(dataset, group):
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    output = model(**model_inputs(model, batch, device))
                part = numerator(output["logit"], batch["label"].to(device),
                                 batch["group_weight"].to(device), settings["positive_weight"])
                if training:
                    scaler.scale(part / denom).backward()
            total += float(part.detach())
        if training:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            if not torch.isfinite(norm) and not amp:
                raise FloatingPointError(f"Non-finite gradient in {group.group_id}")
            scaler.step(optimizer)
            scaler.update()
        losses.append(total / float(denom))
    return float(np.mean(losses))


def optimizer_for(model, settings):
    return torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                             weight_decay=settings["weight_decay"])


def fit_inner(model, train, validation, device, settings):
    optimizer = optimizer_for(model, settings)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
    best_loss, best_epoch, stale = np.inf, 0, 0
    train_at_best = np.nan
    for epoch in range(settings["max_epochs"]):
        train_loss = run_epoch(model, train, optimizer, device, settings, epoch=epoch, scaler=scaler)
        valid_loss = run_epoch(model, validation, None, device, settings)
        if valid_loss < best_loss - settings["minimum_delta"]:
            best_loss, best_epoch, train_at_best, stale = valid_loss, epoch + 1, train_loss, 0
        else:
            stale += 1
        if stale >= settings["early_stopping_patience"]:
            break
    return {"best_epoch": best_epoch, "epochs_run": epoch + 1,
            "train_loss_at_best": train_at_best, "validation_loss_at_best": best_loss}


def refit(model, train, device, settings, seed, epochs):
    optimizer = optimizer_for(model, settings)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
    history = []
    for epoch in range(epochs):
        loss = run_epoch(model, train, optimizer, device, settings,
                         seed=seed, epoch=epoch, scaler=scaler)
        history.append({"epoch": epoch + 1, "train_group_loss": loss})
    return history


@torch.inference_mode()
def predict(models, dataset, device, outer_fold, seeds):
    for model in models:
        model.eval()
    locus, variants = [], []
    for _, matched in dataset.index.groupby("match_set_id", sort=True):
        indices = matched.sort_values("mil_locus_id").index.to_numpy()
        if len(indices) != 3:
            raise ValueError("A test match set must contain three bags")
        batch = collate([dataset[int(i)] for i in indices])
        outputs = [model(**model_inputs(model, batch, device)) for model in models]
        logits = torch.stack([o["logit"] for o in outputs]).float().cpu().numpy()
        variant_scores = torch.stack([o["variant_score"] for o in outputs]).float().cpu().numpy()
        for i, locus_id in enumerate(batch["locus_id"]):
            mean_logit = float(logits[:, i].mean())
            locus.append({"mil_locus_id": locus_id, "outer_fold": outer_fold,
                          **{f"logit_seed_{s}": float(logits[j, i]) for j, s in enumerate(seeds)},
                          "logit": mean_logit,
                          "probability": float(torch.sigmoid(torch.tensor(mean_logit)))})
            n = int(batch["variant_mask"][i].sum())
            for k in range(n):
                values = variant_scores[:, i, k]
                variants.append({"mil_locus_id": locus_id,
                                 "variant_row": int(batch["variant_row"][i, k]),
                                 "outer_fold": outer_fold,
                                 **{f"variant_score_seed_{s}": float(values[j])
                                    for j, s in enumerate(seeds)},
                                 "variant_score": float(values.mean())})
    return pd.DataFrame(locus), pd.DataFrame(variants)
