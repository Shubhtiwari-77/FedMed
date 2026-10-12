"""
src/data_utils.py
=================
Turns a validated hospital DataFrame into PyTorch tensors, and provides the
Non-IID Dirichlet partitioner used for the ablation study.

IMPORTANT FL DETAIL
-------------------
Feature scaling statistics (mean/std) must NOT be computed over the union of
all hospitals — that would be a silent data leak across the privacy boundary.
We therefore use FIXED, clinically-derived normalisation constants published
in `config`-style form below. Every hospital applies the same public constants
locally, so no statistic ever crosses the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import config as cfg

# Public, clinically-derived scaling constants (mid-point, half-range).
# These are textbook reference ranges, not statistics learned from anyone's data.
_SCALE = {
    "age":               (45.0, 45.0),
    "fever_days":        (10.0, 10.0),
    "temp_max_c":        (38.5, 2.5),
    "platelet_count_k":  (250.0, 200.0),
    "wbc_count_k":       (9.0, 8.0),
    "hemoglobin_gdl":    (13.0, 5.0),
    "crp_mgl":           (60.0, 60.0),
    "spo2_pct":          (95.0, 8.0),
    "systolic_bp":       (120.0, 40.0),
}

MISSING_FLAGS = [f"{c}_was_missing" for c in cfg.CONTINUOUS_FEATURES + ["systolic_bp"]]


def feature_names() -> list[str]:
    """Exact, ordered list of model input columns. Order must never change."""
    names = list(_SCALE.keys())
    names += cfg.BINARY_SYMPTOMS
    names += ["gender"]
    names += [f"region_{r}" for r in cfg.REGIONS]
    names += [f"season_{s}" for s in cfg.SEASONS]
    names += ["month_sin", "month_cos"]
    names += MISSING_FLAGS
    return names


NUM_FEATURES = len(feature_names())


def encode(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """DataFrame -> (X float32 [n, d], y int64 [n]) using the fixed schema."""
    df = df.copy()

    for flag in MISSING_FLAGS:
        if flag not in df.columns:
            df[flag] = 0

    cols = {}
    for c, (mid, half) in _SCALE.items():
        cols[c] = (pd.to_numeric(df[c], errors="coerce").fillna(mid) - mid) / half
    for s in cfg.BINARY_SYMPTOMS:
        cols[s] = df[s].astype(float)
    cols["gender"] = df["gender"].astype(float)
    for r in cfg.REGIONS:
        cols[f"region_{r}"] = (df["region"] == r).astype(float)
    for s in cfg.SEASONS:
        cols[f"season_{s}"] = (df["season"] == s).astype(float)
    # Cyclic month encoding so December and January are neighbours
    m = pd.to_numeric(df["month"], errors="coerce").fillna(6)
    cols["month_sin"] = np.sin(2 * np.pi * m / 12)
    cols["month_cos"] = np.cos(2 * np.pi * m / 12)
    for flag in MISSING_FLAGS:
        cols[flag] = df[flag].astype(float)

    X = np.column_stack([np.asarray(cols[n], dtype=np.float32) for n in feature_names()])
    y = df["label"].to_numpy(dtype=np.int64)
    return X.astype(np.float32), y


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = cfg.BATCH_SIZE,
                shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------------------------------------------------------------------------
# NON-IID DIRICHLET PARTITIONER  (for the ablation study in the paper)
# ---------------------------------------------------------------------------
def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float = 0.4,
                        seed: int = cfg.SEED, min_size: int = 20) -> list[np.ndarray]:
    """Split indices across clients with a Dirichlet(alpha) label skew.

    alpha -> 0    = extreme skew (each client sees ~1 class)
    alpha -> inf  = IID (every client sees the same distribution)

    Use this when a reviewer asks "how does your method behave as
    heterogeneity increases?" — sweep alpha in {0.1, 0.5, 1.0, 100}.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1

    while True:
        idx_per_client = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            p = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(p)[:-1] * len(idx_c)).astype(int)
            for i, part in enumerate(np.split(idx_c, cuts)):
                idx_per_client[i].extend(part.tolist())
        sizes = [len(a) for a in idx_per_client]
        if min(sizes) >= min_size:
            break

    return [np.array(sorted(a)) for a in idx_per_client]


def label_distribution_table(splits: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-client class counts — goes straight into the paper as Table 1."""
    rows = {}
    for cid, y in splits.items():
        rows[cid] = pd.Series(y).value_counts().reindex(range(cfg.NUM_CLASSES), fill_value=0)
    out = pd.DataFrame(rows).T
    out.columns = cfg.DISEASES
    out["TOTAL"] = out.sum(axis=1)
    return out
