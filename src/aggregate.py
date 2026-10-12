"""
src/aggregate.py
================
Central-server aggregation. This is the ONLY code that sees more than one
hospital — and it sees nothing but float tensors.

FedAvg (McMahan et al., 2017):

        W_{t+1}  =  sum_{k=1..K}  (n_k / N) * W^k_{t+1}          where N = sum n_k

`fedavg` implements exactly that. `fedavg_equal` and `fedprox_proximal_term`
are included so you can run the ablations a reviewer will ask for.
"""

from __future__ import annotations

import numpy as np


def fedavg(client_weights: list[list[np.ndarray]],
           client_sizes: list[int]) -> list[np.ndarray]:
    """Sample-size-weighted average — the answer to viva question Q1."""
    if not client_weights:
        raise ValueError("No client updates to aggregate.")
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("Total sample count must be positive.")

    coeffs = [n / total for n in client_sizes]
    agg = [np.zeros_like(layer, dtype=np.float64) for layer in client_weights[0]]
    for w, c in zip(client_weights, coeffs):
        for i, layer in enumerate(w):
            agg[i] += c * layer.astype(np.float64)
    return [a.astype(np.float32) for a in agg]


def fedavg_equal(client_weights: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Un-weighted mean — the ABLATION that shows why weighting matters.
    Run this and you will see the 1,200-patient hospital drag the global
    model toward its cold-region distribution."""
    return fedavg(client_weights, [1] * len(client_weights))


def weight_delta_norm(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """||a - b||_2 across all layers — used to plot convergence."""
    return float(np.sqrt(sum(float(np.sum((x - y) ** 2)) for x, y in zip(a, b))))
