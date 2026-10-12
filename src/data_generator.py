"""
src/data_generator.py
=====================
Generates a realistic, context-aware synthetic EHR dataset.

WHY SYNTHETIC?
--------------
MIMIC-IV / ChestX-ray14 need credentialed access + 100s of GB and cannot be
run on two student laptops. This module instead implements an explicit
*Bayesian generative process* that encodes real epidemiological priors:

        P(disease | region, season)  x  P(symptoms, labs | disease)

That makes the "context matters" claim in the proposal mathematically
concrete and defensible in a viva, and it lets you prove the model learned
the geography/season effect (a real dataset would only let you assert it).

Swap in a real dataset later by replacing `build_hospital_dataframe()` with a
loader that returns the same column schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


# ---------------------------------------------------------------------------
# 1. EPIDEMIOLOGICAL PRIOR:  P(disease | region, season)
# ---------------------------------------------------------------------------
# Rows = region, Cols = disease, one matrix per season. Values are unnormalised
# relative risks; they get normalised into a probability vector at sample time.
#           URI  Flu  Deng Chik Mal  Typh Pneu Auto
_PRIOR = {
    "Monsoon": {
        "Tropical_Monsoon": [18, 6, 22, 14, 16, 14, 6, 4],
        "Tropical_Dry":     [20, 7, 12, 8, 10, 16, 7, 5],
        "Temperate":        [26, 12, 6, 4, 4, 10, 10, 8],
        "Cold_Subzero":     [30, 18, 0, 0, 0, 5, 20, 17],   # no vector-borne
    },
    "Post_Monsoon": {
        "Tropical_Monsoon": [18, 9, 20, 12, 14, 12, 8, 5],
        "Tropical_Dry":     [22, 10, 11, 7, 9, 14, 10, 6],
        "Temperate":        [26, 15, 5, 3, 3, 9, 14, 9],
        "Cold_Subzero":     [28, 22, 0, 0, 0, 4, 26, 18],
    },
    "Winter": {
        "Tropical_Monsoon": [26, 20, 5, 3, 4, 8, 22, 10],
        "Tropical_Dry":     [27, 20, 3, 2, 3, 9, 22, 11],
        "Temperate":        [24, 24, 1, 1, 1, 6, 28, 13],
        "Cold_Subzero":     [22, 26, 0, 0, 0, 3, 32, 22],   # pneumonia + autoimmune peak
    },
    "Summer": {
        "Tropical_Monsoon": [24, 8, 8, 5, 7, 22, 8, 7],
        "Tropical_Dry":     [24, 8, 5, 3, 5, 26, 8, 8],
        "Temperate":        [28, 12, 3, 2, 2, 16, 12, 10],
        "Cold_Subzero":     [32, 14, 0, 0, 0, 8, 18, 18],
    },
}

_MONTH_WEIGHTS = {
    "uniform":      np.ones(12),
    "monsoon_heavy": np.array([0.4, 0.4, 0.5, 0.6, 0.8, 1.8, 2.4, 2.6, 2.0, 1.4, 0.9, 0.5]),
    "winter_heavy":  np.array([2.4, 2.2, 1.4, 0.8, 0.6, 0.5, 0.5, 0.5, 0.7, 1.1, 1.8, 2.3]),
}


# ---------------------------------------------------------------------------
# 2. CLASS-CONDITIONAL CLINICAL PROFILES:  P(features | disease)
# ---------------------------------------------------------------------------
# Continuous features: (mean, std). Binary symptoms: probability of presence.
# Values follow standard clinical presentations (Harrison's / WHO case defs).
_PROFILE = {
    "Viral_URI": {
        "cont": {"fever_days": (2.5, 1.2), "temp_max_c": (38.1, 0.5), "platelet_count_k": (255, 55),
                 "wbc_count_k": (7.5, 2.0), "hemoglobin_gdl": (13.2, 1.3), "crp_mgl": (9, 7), "spo2_pct": (98, 1.0)},
        "sym": {"joint_pain": .10, "rash": .04, "retro_orbital_pain": .05, "headache": .45, "cough": .70,
                "sore_throat": .80, "breathlessness": .05, "vomiting": .08, "abdominal_pain": .06, "chills_rigor": .15},
        "age": (28, 16),
    },
    "Influenza": {
        "cont": {"fever_days": (4.0, 1.6), "temp_max_c": (39.2, 0.6), "platelet_count_k": (210, 55),
                 "wbc_count_k": (5.2, 1.8), "hemoglobin_gdl": (13.0, 1.4), "crp_mgl": (28, 16), "spo2_pct": (96, 1.8)},
        "sym": {"joint_pain": .45, "rash": .05, "retro_orbital_pain": .20, "headache": .78, "cough": .82,
                "sore_throat": .60, "breathlessness": .18, "vomiting": .15, "abdominal_pain": .10, "chills_rigor": .60},
        "age": (33, 19),
    },
    "Dengue": {
        "cont": {"fever_days": (6.0, 2.0), "temp_max_c": (39.6, 0.6), "platelet_count_k": (78, 34),   # <- thrombocytopenia
                 "wbc_count_k": (3.1, 1.1), "hemoglobin_gdl": (14.4, 1.7), "crp_mgl": (24, 14), "spo2_pct": (97, 1.6)},
        "sym": {"joint_pain": .72, "rash": .62, "retro_orbital_pain": .80, "headache": .88, "cough": .18,
                "sore_throat": .14, "breathlessness": .12, "vomiting": .55, "abdominal_pain": .48, "chills_rigor": .50},
        "age": (30, 15),
    },
    "Chikungunya": {
        "cont": {"fever_days": (5.2, 1.8), "temp_max_c": (39.1, 0.6), "platelet_count_k": (172, 48),
                 "wbc_count_k": (4.4, 1.5), "hemoglobin_gdl": (13.6, 1.4), "crp_mgl": (40, 20), "spo2_pct": (98, 1.2)},
        "sym": {"joint_pain": .95, "rash": .48, "retro_orbital_pain": .25, "headache": .70, "cough": .12,
                "sore_throat": .12, "breathlessness": .06, "vomiting": .28, "abdominal_pain": .22, "chills_rigor": .42},
        "age": (36, 17),
    },
    "Malaria": {
        "cont": {"fever_days": (7.0, 3.0), "temp_max_c": (39.8, 0.7), "platelet_count_k": (105, 42),
                 "wbc_count_k": (5.0, 1.9), "hemoglobin_gdl": (10.4, 1.8), "crp_mgl": (62, 28), "spo2_pct": (96, 2.0)},
        "sym": {"joint_pain": .30, "rash": .05, "retro_orbital_pain": .10, "headache": .74, "cough": .14,
                "sore_throat": .08, "breathlessness": .20, "vomiting": .52, "abdominal_pain": .40, "chills_rigor": .92},
        "age": (31, 17),
    },
    "Typhoid": {
        "cont": {"fever_days": (9.5, 3.5), "temp_max_c": (39.4, 0.6), "platelet_count_k": (162, 50),
                 "wbc_count_k": (3.9, 1.4), "hemoglobin_gdl": (11.6, 1.6), "crp_mgl": (70, 30), "spo2_pct": (97, 1.5)},
        "sym": {"joint_pain": .20, "rash": .18, "retro_orbital_pain": .06, "headache": .66, "cough": .28,
                "sore_throat": .16, "breathlessness": .10, "vomiting": .44, "abdominal_pain": .78, "chills_rigor": .48},
        "age": (24, 13),
    },
    "Bacterial_Pneumonia": {
        "cont": {"fever_days": (5.0, 2.2), "temp_max_c": (39.3, 0.8), "platelet_count_k": (330, 90),
                 "wbc_count_k": (16.5, 4.5), "hemoglobin_gdl": (12.2, 1.6), "crp_mgl": (145, 55), "spo2_pct": (91, 3.4)},
        "sym": {"joint_pain": .14, "rash": .03, "retro_orbital_pain": .04, "headache": .40, "cough": .93,
                "sore_throat": .26, "breathlessness": .86, "vomiting": .18, "abdominal_pain": .14, "chills_rigor": .70},
        "age": (52, 20),
    },
    "Autoimmune_Flare": {
        "cont": {"fever_days": (14.0, 7.0), "temp_max_c": (38.0, 0.7), "platelet_count_k": (290, 85),
                 "wbc_count_k": (9.0, 3.4), "hemoglobin_gdl": (10.8, 1.7), "crp_mgl": (55, 30), "spo2_pct": (97, 1.8)},
        "sym": {"joint_pain": .90, "rash": .42, "retro_orbital_pain": .08, "headache": .38, "cough": .18,
                "sore_throat": .10, "breathlessness": .22, "vomiting": .12, "abdominal_pain": .24, "chills_rigor": .20},
        "age": (44, 15),
    },
}


# ---------------------------------------------------------------------------
# 3. SAMPLING
# ---------------------------------------------------------------------------
def _sample_disease(rng, region: str, season: str) -> int:
    w = np.asarray(_PRIOR[season][region], dtype=float)
    return int(rng.choice(cfg.NUM_CLASSES, p=w / w.sum()))


def _sample_case(rng, disease_idx: int) -> dict:
    name = cfg.DISEASES[disease_idx]
    prof = _PROFILE[name]
    row = {}
    for feat, (mu, sd) in prof["cont"].items():
        lo, hi = cfg.PHYSIOLOGICAL_BOUNDS[feat]
        row[feat] = float(np.clip(rng.normal(mu, sd), lo, hi))
    a_mu, a_sd = prof["age"]
    row["age"] = float(np.clip(rng.normal(a_mu, a_sd), 1, 95))
    row["fever_days"] = float(max(0.0, round(row["fever_days"])))
    for sym, p in prof["sym"].items():
        row[sym] = int(rng.random() < p)
    row["systolic_bp"] = float(np.clip(rng.normal(118, 14), 70, 200))
    return row


def build_hospital_dataframe(
    n_samples: int,
    region_mix: dict,
    month_mix: str = "uniform",
    label_noise: float = 0.05,
    missing_rate: float = 0.05,
    corrupt_rate: float = 0.02,
    seed: int = 0,
) -> pd.DataFrame:
    """Create one hospital's raw, *uncleaned* admission log.

    Deliberately injects four real-world defects so the Phase-2 validation
    layer has something genuine to catch:
      1. `label_noise`   -> provisional diagnosis later overturned
      2. `missing_rate`  -> lab result not yet back
      3. `corrupt_rate`  -> data-entry errors (age 250, BP 900)
      4. quarantine/ambiguity flags -> Phase-1 holding buffer
    """
    rng = np.random.default_rng(seed)

    regions = list(region_mix.keys())
    rprob = np.array([region_mix[r] for r in regions], dtype=float)
    rprob = rprob / rprob.sum()

    mw = _MONTH_WEIGHTS[month_mix].astype(float)
    mw = mw / mw.sum()

    rows = []
    for _ in range(n_samples):
        region = str(rng.choice(regions, p=rprob))
        month = int(rng.choice(np.arange(1, 13), p=mw))
        season = cfg.MONTH_TO_SEASON[month]

        true_label = _sample_disease(rng, region, season)
        row = _sample_case(rng, true_label)

        row["region"] = region
        row["month"] = month
        row["season"] = season
        row["gender"] = int(rng.random() < 0.52)          # 1 = male
        row["true_label"] = true_label

        # --- Phase-1 clinical workflow metadata -------------------------
        row["days_since_admission"] = int(rng.integers(1, 45))
        conf = float(np.clip(rng.beta(6, 2), 0, 1))       # clinician confidence
        row["clinician_confidence"] = conf
        if conf < cfg.AMBIGUITY_THRESHOLD:
            row["diagnosis_status"] = "AMBIGUOUS"
        elif row["days_since_admission"] < cfg.QUARANTINE_DAYS:
            row["diagnosis_status"] = "PENDING"
        else:
            row["diagnosis_status"] = "CONFIRMED"

        # --- defect 1: provisional label later overturned ----------------
        recorded = true_label
        if rng.random() < label_noise:
            recorded = int(rng.integers(0, cfg.NUM_CLASSES))
        row["label"] = recorded
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- defect 2: missing lab values -----------------------------------
    for col in ["platelet_count_k", "wbc_count_k", "crp_mgl", "hemoglobin_gdl", "spo2_pct"]:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    # --- defect 3: data-entry corruption --------------------------------
    n_bad = max(1, int(corrupt_rate * len(df)))
    bad_idx = rng.choice(len(df), size=n_bad, replace=False)
    for i in bad_idx:
        which = rng.integers(0, 3)
        if which == 0:
            df.loc[df.index[i], "age"] = float(rng.integers(150, 300))
        elif which == 1:
            df.loc[df.index[i], "systolic_bp"] = float(rng.integers(400, 900))
        else:
            df.loc[df.index[i], "temp_max_c"] = float(rng.integers(60, 120))

    return df


def build_all(seed: int = cfg.SEED) -> dict:
    """Generate every hospital's raw log + the neutral global test set."""
    out = {}
    for i, (hid, spec) in enumerate(cfg.HOSPITALS.items()):
        out[hid] = build_hospital_dataframe(
            n_samples=spec["n_samples"],
            region_mix=spec["region_mix"],
            month_mix=spec["month_mix"],
            label_noise=spec["label_noise"],
            missing_rate=spec["missing_rate"],
            seed=seed + i,
        )

    # Global test set: clean, all regions equally -> measures true generalisation
    out["GLOBAL_TEST"] = build_hospital_dataframe(
        n_samples=cfg.GLOBAL_TEST_SIZE,
        region_mix={r: 0.25 for r in cfg.REGIONS},
        month_mix="uniform",
        label_noise=0.0,          # test labels must be ground truth
        missing_rate=0.02,
        corrupt_rate=0.0,
        seed=seed + 999,
    )
    return out


if __name__ == "__main__":
    import os

    np.random.seed(cfg.SEED)
    data = build_all()
    for name, df in data.items():
        path = os.path.join(cfg.DATA_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        dist = df["label"].value_counts().reindex(range(cfg.NUM_CLASSES), fill_value=0)
        print(f"{name:20s} n={len(df):5d} -> saved {path}")
        print("   label dist:", dist.to_list())
