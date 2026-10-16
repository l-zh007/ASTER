"""Published dependency-group weighted binary cross-entropy."""

import torch
import torch.nn.functional as F


def numerator(logits, labels, weights, positive_weight=2.0):
    logits, labels, weights = logits.float(), labels.float(), weights.float()
    per_bag = (1 - labels) * F.softplus(logits) + positive_weight * labels * F.softplus(-logits)
    return (weights * per_bag).sum()
