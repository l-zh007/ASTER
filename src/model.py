"""Final Clean-G0 model; no RNA/Gene branch is used in the forward pass."""

from __future__ import annotations

import torch
from torch import nn


def initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class NormalizedLSE(nn.Module):
    def __init__(self, tau: float):
        super().__init__()
        self.tau = tau

    def forward(self, scores: torch.Tensor, mask: torch.Tensor, *, nonempty=False):
        valid = mask.any(dim=-1)
        if nonempty and not bool(valid.all()):
            raise ValueError("A locus has no valid variant–cell instances")
        scaled = (scores.float() / self.tau).masked_fill(~mask, -torch.inf)
        scaled = torch.where(valid.unsqueeze(-1), scaled, torch.zeros_like(scaled))
        pooled = self.tau * (
            torch.logsumexp(scaled, dim=-1) - torch.log(mask.sum(-1).clamp_min(1))
        )
        return torch.where(valid, pooled, torch.zeros_like(pooled)).to(scores.dtype), valid


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, seq, mask):
        return self.projection(seq) * mask.unsqueeze(-1)


class ATACEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.GELU(),
        )

    def forward(self, atac, mask):
        b, m, c, bins, channels = atac.shape
        flat = atac.reshape(b * m * c, bins, channels).transpose(1, 2)
        valid_rows = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if not valid_rows.numel():
            return atac.new_zeros((b, m, c, 64))
        encoded = self.convolution(flat.index_select(0, valid_rows))
        pooled = torch.cat([encoded.mean(-1), encoded.amax(-1)], dim=-1)
        hidden = pooled.new_zeros((b * m * c, 64)).index_copy(0, valid_rows, pooled)
        return hidden.reshape(b, m, c, 64)


class ResidualGate(nn.Module):
    def __init__(self, sequence_dim, atac_dim):
        super().__init__()
        self.sequence_query = nn.Linear(sequence_dim, sequence_dim, bias=False)
        self.atac_gate = nn.Linear(atac_dim, sequence_dim)
        self.normalization = nn.LayerNorm(sequence_dim)

    def forward(self, sequence, atac, sequence_mask):
        mask = sequence_mask.unsqueeze(-1).to(sequence.dtype)
        query = self.sequence_query(sequence) * mask
        gate = torch.sigmoid(self.atac_gate(atac))
        residual = query.unsqueeze(2).expand_as(gate)
        regulatory = self.normalization(residual + residual * gate)
        return regulatory * mask.unsqueeze(2), gate


class Fusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class ASTER(nn.Module):
    input_keys = ("seq", "sequence_mask", "atac", "variant_mask", "cell_mask", "trait_id")

    def __init__(self, config: dict, dropout: float):
        super().__init__()
        self.config = dict(config)
        seq_hidden = config["sequence_hidden"]
        atac_hidden = config["atac_hidden"]
        self.sequence_encoder = SequenceEncoder(config["sequence_features"], seq_hidden, dropout)
        self.atac_encoder = ATACEncoder(config["atac_channels"])
        self.regulatory_gate = ResidualGate(seq_hidden, atac_hidden)
        self.fusion = Fusion(seq_hidden + atac_hidden, config["fusion_hidden"], dropout)
        self.instance_scorer = nn.Linear(config["fusion_hidden"], 1)
        self.cell_pool = NormalizedLSE(config["tau_cell"])
        self.variant_pool = NormalizedLSE(config["tau_variant"])
        self.apply(initialize)

    def forward(self, *, seq, sequence_mask, atac, variant_mask, cell_mask, trait_id):
        if torch.any(trait_id != 0):
            raise ValueError("ASTER is configured for single-trait T2D")
        sequence_mask = sequence_mask & variant_mask
        cell_mask = cell_mask & variant_mask.unsqueeze(-1)
        sequence_hidden = self.sequence_encoder(seq, sequence_mask)
        atac_hidden = self.atac_encoder(atac, cell_mask)
        regulatory, gate = self.regulatory_gate(sequence_hidden, atac_hidden, sequence_mask)
        fused = self.fusion(torch.cat([regulatory, atac_hidden], dim=-1))
        cell_scores = self.instance_scorer(fused).squeeze(-1)
        cell_scores = torch.where(cell_mask, cell_scores, torch.zeros_like(cell_scores))
        variant_scores, cell_valid = self.cell_pool(cell_scores, cell_mask)
        effective_mask = variant_mask & cell_valid
        locus_logits, _ = self.variant_pool(variant_scores, effective_mask, nonempty=True)
        return {
            "logit": locus_logits,
            "variant_score": variant_scores,
            "variant_cell_score": cell_scores,
            "atac_gate": gate,
            "effective_variant_mask": effective_mask,
        }


class SequenceOnly(nn.Module):
    """Paper control: identical Sequence projection and variant LSE, without ATAC."""

    input_keys = ("seq", "sequence_mask", "variant_mask")

    def __init__(self, config: dict, dropout: float = 0.15):
        super().__init__()
        self.sequence_encoder = SequenceEncoder(
            config["sequence_features"], config["sequence_hidden"], dropout)
        self.variant_scorer = nn.Linear(config["sequence_hidden"], 1)
        self.variant_pool = NormalizedLSE(config["tau_variant"])
        self.apply(initialize)

    def forward(self, *, seq, sequence_mask, variant_mask):
        valid = variant_mask & sequence_mask
        hidden = self.sequence_encoder(seq, valid)
        scores = self.variant_scorer(hidden).squeeze(-1)
        scores = torch.where(valid, scores, torch.zeros_like(scores))
        logit, _ = self.variant_pool(scores, valid, nonempty=True)
        return {"logit": logit, "variant_score": scores,
                "effective_variant_mask": valid}
