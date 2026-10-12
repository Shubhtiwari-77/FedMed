"""
src/model.py
============
`DiagnosisNet` — the local model every hospital trains, and the global model
the server aggregates. A Multi-Layer Perceptron is the correct architecture
here because the input is structured tabular EHR data, not imagery.

(If you later swap to ChestX-ray14, replace this class with a torchvision
ResNet-18 and change nothing else — the FL code is architecture-agnostic
because it only ever touches `get_weights()` / `set_weights()`.)
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import config as cfg


class DiagnosisNet(nn.Module):
    def __init__(self, in_dim: int, num_classes: int | None = None,
                 hidden: list[int] | None = None, dropout: float | None = None):
        super().__init__()
        # Resolved at call time, NOT as a default argument: Kaggle mode
        # rewrites cfg.NUM_CLASSES at runtime, and a default argument would
        # have been frozen to 8 at import.
        num_classes = num_classes if num_classes is not None else cfg.NUM_CLASSES
        dropout = dropout if dropout is not None else cfg.DROPOUT
        hidden = hidden or cfg.HIDDEN_DIMS
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                # LayerNorm, not BatchNorm: BatchNorm running stats are data-
                # dependent and would leak cohort statistics through the weights.
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# WEIGHT SERIALISATION — the ONLY thing that ever leaves a hospital
# ---------------------------------------------------------------------------
def get_weights(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = OrderedDict(
        {k: torch.tensor(np.asarray(w), dtype=torch.float32) for k, w in zip(keys, weights)}
    )
    model.load_state_dict(state, strict=True)


def build_model(in_dim: int, seed: int = cfg.SEED,
                num_classes: int | None = None) -> DiagnosisNet:
    torch.manual_seed(seed)
    return DiagnosisNet(in_dim, num_classes=num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
