"""
config.py
=========
Single source of truth for every hyper-parameter, path and clinical constant
used by the FedMed-FL project.  Change values HERE, never inside the scripts —
that is what makes the ablation studies in Section 7 of the proposal easy.
"""

import os

# ----------------------------------------------------------------------------
# 0. PATHS
# ----------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42

# ----------------------------------------------------------------------------
# 1. CLINICAL ONTOLOGY  (the 8 target classes)
# ----------------------------------------------------------------------------
DISEASES = [
    "Viral_URI",             # 0  common viral upper-respiratory infection
    "Influenza",             # 1
    "Dengue",                # 2  vector-borne, monsoon, tropical
    "Chikungunya",           # 3  vector-borne, monsoon, tropical
    "Malaria",               # 4  vector-borne, monsoon/post-monsoon
    "Typhoid",               # 5  water-borne, monsoon/summer
    "Bacterial_Pneumonia",   # 6  winter-skewed, all regions
    "Autoimmune_Flare",      # 7  region-independent, cold-region differential
]
NUM_CLASSES = len(DISEASES)

REGIONS = ["Tropical_Monsoon", "Tropical_Dry", "Temperate", "Cold_Subzero"]
SEASONS = ["Winter", "Summer", "Monsoon", "Post_Monsoon"]

# Month (1-12) -> season, used by the context-aware inference engine
MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post_Monsoon", 11: "Post_Monsoon",
}

BINARY_SYMPTOMS = [
    "joint_pain", "rash", "retro_orbital_pain", "headache", "cough",
    "sore_throat", "breathlessness", "vomiting", "abdominal_pain", "chills_rigor",
]

CONTINUOUS_FEATURES = [
    "age", "fever_days", "temp_max_c", "platelet_count_k",
    "wbc_count_k", "hemoglobin_gdl", "crp_mgl", "spo2_pct",
]

# Physiologically possible ranges — used by the Phase-2 validation quality gate
PHYSIOLOGICAL_BOUNDS = {
    "age": (0, 120),
    "fever_days": (0, 60),
    "temp_max_c": (34.0, 43.0),
    "platelet_count_k": (5, 800),        # x10^3 /uL
    "wbc_count_k": (0.5, 60.0),          # x10^3 /uL
    "hemoglobin_gdl": (3.0, 20.0),
    "crp_mgl": (0.0, 400.0),
    "spo2_pct": (50, 100),
    "systolic_bp": (60, 250),
}

# ----------------------------------------------------------------------------
# 2. HOSPITAL (CLIENT) PROFILES  ->  source of natural Non-IID skew
# ----------------------------------------------------------------------------
# Each hospital sits in a different climate zone and sees a different
# admission volume. This produces label skew WITHOUT any artificial trick,
# which is exactly the "institutional variance" argued for in the proposal.
HOSPITALS = {
    "H1_Max_Delhi": {
        "n_samples": 4000,
        "region_mix": {"Tropical_Monsoon": 0.80, "Temperate": 0.15, "Tropical_Dry": 0.05, "Cold_Subzero": 0.00},
        "month_mix": "monsoon_heavy",
        "label_noise": 0.05,
        "missing_rate": 0.04,
    },
    "H2_Apollo_Chennai": {
        "n_samples": 3000,
        "region_mix": {"Tropical_Monsoon": 0.45, "Tropical_Dry": 0.50, "Temperate": 0.05, "Cold_Subzero": 0.00},
        "month_mix": "uniform",
        "label_noise": 0.06,
        "missing_rate": 0.06,
    },
    "H3_SKIMS_Srinagar": {
        "n_samples": 1200,          # deliberately small -> tests FedAvg weighting
        "region_mix": {"Cold_Subzero": 0.75, "Temperate": 0.25, "Tropical_Monsoon": 0.00, "Tropical_Dry": 0.00},
        "month_mix": "winter_heavy",
        "label_noise": 0.04,
        "missing_rate": 0.03,
    },
}
CLIENT_IDS = list(HOSPITALS.keys())
NUM_CLIENTS = len(CLIENT_IDS)

# Held-out multi-region test set owned by nobody (used only for global eval)
GLOBAL_TEST_SIZE = 2400
VAL_FRACTION = 0.30             # of the hold-out: used to PICK the round, never reported

# ----------------------------------------------------------------------------
# 3. PHASE-1 / PHASE-2  VALIDATION PIPELINE
# ----------------------------------------------------------------------------
QUARANTINE_DAYS = 14            # 1-2 week holding buffer before a case is trainable
AMBIGUITY_THRESHOLD = 0.60      # clinician confidence below this -> not trainable
LOSS_FILTER_PERCENTILE = 95     # drop the top 5% highest-loss (contradictory) samples
LOSS_FILTER_START_ROUND = 3     # only start loss-filtering once the model is sane

# ----------------------------------------------------------------------------
# 4. MODEL
# ----------------------------------------------------------------------------
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.30

# ----------------------------------------------------------------------------
# 5. FEDERATED TRAINING
# ----------------------------------------------------------------------------
NUM_ROUNDS = 20                 # global communication rounds
LOCAL_EPOCHS = 3                # E — local epochs per round
BATCH_SIZE = 64
BATCH_SIZE_DP = 32              # smaller batch -> smaller q -> much better epsilon
LEARNING_RATE = 1e-3
LEARNING_RATE_DP = 8e-3         # DP-SGD needs a larger LR (gradients are clipped to C)
WEIGHT_DECAY = 1e-4

# ----------------------------------------------------------------------------
# 6. DIFFERENTIAL PRIVACY  (client-side / local DP on the model delta)
# ----------------------------------------------------------------------------
# "record" = DP-SGD per patient record (DEFAULT — protects a patient, HIPAA-relevant)
# "client" = clip+noise the transmitted update (protects a hospital; ablation only)
# "none"   = no DP (utility-ceiling baseline)
DP_MODE = "record"
DP_CLIP_NORM = 1.0              # C — per-example gradient L2 clipping bound
DP_NOISE_MULTIPLIER = 1.50      # sigma — noise std = sigma * C  (eps ~ 6 at T=30)
DP_DELTA = 1e-5                 # target delta for the (eps, delta) report

# ----------------------------------------------------------------------------
# 7. CENTRALIZED BASELINE (for the comparison table in the paper)
# ----------------------------------------------------------------------------
CENTRAL_EPOCHS = NUM_ROUNDS * LOCAL_EPOCHS   # equal compute budget = fair comparison

# ----------------------------------------------------------------------------
# 8. FLOWER (real 2-laptop deployment)
# ----------------------------------------------------------------------------
SERVER_ADDRESS = "0.0.0.0:8080"          # what the SERVER binds to
CLIENT_CONNECT_ADDRESS = "127.0.0.1:8080"  # what the CLIENT dials -> change to 192.168.x.x:8080
