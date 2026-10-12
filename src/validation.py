"""
src/validation.py
=================
Implements Phase-1 (Quarantine / Holding Buffer) and Phase-2 (Local Data
Validation Quality Gate) from the proposal. Runs ENTIRELY inside the hospital
boundary — nothing here ever touches the network.

Pipeline order (each stage returns a report dict for the paper's Table 2):

    raw admission log
        -> [1] quarantine gate        (unresolved / ambiguous cases held back)
        -> [2] schema + type check    (wrong dtypes, out-of-vocabulary values)
        -> [3] physiological bounds   (age=250, BP=900, temp=95C -> dropped)
        -> [4] missing value handling (median imputation + missingness flags)
        -> [5] loss-based label filter (applied later, during training)
        =  Verified Dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


# ---------------------------------------------------------------------------
# [1] PHASE-1 : QUARANTINE / HOLDING BUFFER
# ---------------------------------------------------------------------------
def quarantine_gate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Only clinically RESOLVED cases may enter the training pool.

    A case leaves quarantine when BOTH hold:
      * >= QUARANTINE_DAYS have elapsed since admission (lab/serology back), and
      * clinician confidence >= AMBIGUITY_THRESHOLD.
    """
    n0 = len(df)
    resolved = (
        (df["days_since_admission"] >= cfg.QUARANTINE_DAYS)
        & (df["clinician_confidence"] >= cfg.AMBIGUITY_THRESHOLD)
        & (df["diagnosis_status"] == "CONFIRMED")
    )
    kept = df[resolved].copy()
    report = {
        "stage": "quarantine_gate",
        "in": n0,
        "out": len(kept),
        "held_pending": int(((df["diagnosis_status"] == "PENDING")).sum()),
        "held_ambiguous": int((df["diagnosis_status"] == "AMBIGUOUS").sum()),
    }
    return kept, report


# ---------------------------------------------------------------------------
# [2] PHASE-2a : SCHEMA INTEGRITY
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = (
    cfg.CONTINUOUS_FEATURES
    + cfg.BINARY_SYMPTOMS
    + ["systolic_bp", "region", "season", "month", "gender", "label"]
)


def schema_check(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Great-Expectations-style assertions, implemented dependency-free."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema violation — missing columns: {missing_cols}")

    n0 = len(df)
    ok = pd.Series(True, index=df.index)
    ok &= df["region"].isin(cfg.REGIONS)
    ok &= df["season"].isin(cfg.SEASONS)
    ok &= df["month"].between(1, 12)
    ok &= df["gender"].isin([0, 1])
    ok &= df["label"].between(0, cfg.NUM_CLASSES - 1)
    for s in cfg.BINARY_SYMPTOMS:
        ok &= df[s].isin([0, 1])

    kept = df[ok].copy()
    return kept, {"stage": "schema_check", "in": n0, "out": len(kept),
                  "dropped": int(n0 - len(kept))}


# ---------------------------------------------------------------------------
# [3] PHASE-2b : PHYSIOLOGICAL BOUNDS + STATISTICAL OUTLIERS
# ---------------------------------------------------------------------------
def bounds_and_outlier_filter(df: pd.DataFrame, z_thresh: float = 8.0
                              ) -> tuple[pd.DataFrame, dict]:
    """Hard physiological bounds, then a robust (MAD-based) z-score screen.

    `z_thresh` is deliberately loose (8 MAD-sigmas). The goal is to remove
    data-entry corruption, NOT rare-but-real presentations — an aggressive
    threshold would silently delete exactly the rare cases the whole federated
    design exists to learn. Lower it only for an ablation study.
    """
    n0 = len(df)
    ok = pd.Series(True, index=df.index)

    # --- hard clinical impossibility -----------------------------------
    for col, (lo, hi) in cfg.PHYSIOLOGICAL_BOUNDS.items():
        if col in df.columns:
            v = df[col]
            ok &= v.isna() | v.between(lo, hi)
    n_impossible = int((~ok).sum())

    # --- robust statistical outliers (modified z-score using MAD) -------
    for col in cfg.CONTINUOUS_FEATURES:
        v = pd.to_numeric(df[col], errors="coerce")
        med = v.median()
        mad = (v - med).abs().median()
        if mad and mad > 0:
            mz = 0.6745 * (v - med).abs() / mad
            ok &= mz.isna() | (mz < z_thresh)

    kept = df[ok].copy()
    return kept, {"stage": "bounds_outlier", "in": n0, "out": len(kept),
                  "impossible": n_impossible, "dropped": int(n0 - len(kept))}


# ---------------------------------------------------------------------------
# [4] PHASE-2c : MISSING VALUE HANDLING
# ---------------------------------------------------------------------------
def impute_missing(df: pd.DataFrame, medians: dict | None = None
                   ) -> tuple[pd.DataFrame, dict, dict]:
    """Median-impute labs and add an explicit `<col>_was_missing` indicator.

    The indicator matters clinically: "platelet count not ordered" is itself
    informative, so throwing that signal away would hurt accuracy.
    Medians are computed LOCALLY (never shared) to keep the FL guarantee.
    """
    out = df.copy()
    n_missing_total = 0
    learned = {}
    for col in cfg.CONTINUOUS_FEATURES + ["systolic_bp"]:
        v = pd.to_numeric(out[col], errors="coerce")
        miss = v.isna()
        n_missing_total += int(miss.sum())
        out[f"{col}_was_missing"] = miss.astype(int)
        med = (medians or {}).get(col, float(v.median()))
        learned[col] = med
        out[col] = v.fillna(med)
    return out, {"stage": "impute", "cells_imputed": n_missing_total}, learned


# ---------------------------------------------------------------------------
# [5] PHASE-2d : LOSS-BASED LABEL FILTER  (used during training)
# ---------------------------------------------------------------------------
def loss_based_mask(per_sample_loss: np.ndarray,
                    percentile: int = cfg.LOSS_FILTER_PERCENTILE) -> np.ndarray:
    """Return a boolean keep-mask that drops the highest-loss samples.

    Rationale: after a few rounds the global model is broadly correct, so a
    sample the model finds *extremely* surprising is far more likely to be a
    mislabelled record than a rare true case. Dropping the top 5% stops those
    records from poisoning the local gradient before it is ever transmitted.
    """
    if len(per_sample_loss) == 0:
        return np.zeros(0, dtype=bool)
    cutoff = np.percentile(per_sample_loss, percentile)
    return per_sample_loss <= cutoff


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------
def run_validation_pipeline(df: pd.DataFrame, client_id: str = "",
                            verbose: bool = True) -> tuple[pd.DataFrame, list]:
    """Full Phase-1 + Phase-2 gate. Returns (verified_df, list_of_reports)."""
    reports = []
    df, r = quarantine_gate(df);            reports.append(r)
    df, r = schema_check(df);               reports.append(r)
    df, r = bounds_and_outlier_filter(df);  reports.append(r)
    df, r, _ = impute_missing(df);          reports.append(r)

    if verbose:
        print(f"\n[VALIDATION] {client_id}")
        for rep in reports:
            extras = {k: v for k, v in rep.items() if k not in ("stage", "in", "out")}
            io = f"{rep.get('in', '-'):>6} -> {rep.get('out', '-'):>6}" if "in" in rep else " " * 16
            print(f"   {rep['stage']:<18} {io}   {extras}")
    return df.reset_index(drop=True), reports


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_generator import build_all

    for cid, raw in build_all().items():
        run_validation_pipeline(raw, cid)
