"""
src/kaggle_data.py
==================
Load a REAL Kaggle symptom->disease dataset and shape it into the same
3-hospital federated setup as the synthetic pipeline.

READ THIS BEFORE YOU PICK A DATASET
-----------------------------------
The most-cited Kaggle disease dataset,
`kaushil268/disease-prediction-using-machine-learning`, is a **deterministic
rule table, not clinical data**. Measured on the real file:

    4,920 rows  ->  only  304  unique symptom patterns
    41 diseases ->  5-10 patterns each, repeated 12-24 times
    0 of 304 patterns map to more than one disease
    plain logistic regression test accuracy: 1.0000

Every model scores 100%. If you train your federated, centralized, isolated,
DP and no-DP runs on it raw, all five will read ~1.00 and your results section
will prove nothing. An examiner who plots a confusion matrix with a perfect
diagonal will ask why, and "the dataset is separable" is not a good answer to
give for the first time in the room.

Two ways to deal with it, both supported here:

  1. PREFERRED — use a bigger, genuinely noisy dataset:
         DATASETS["large"]  ->  dhivyeshrk/diseases-and-symptoms-dataset
         (~246k rows, 773 diseases, 377 symptoms)
     Restrict to the top-K most frequent diseases with `top_k_classes`.

  2. Use the classic dataset but pass it through `inject_ehr_realism()`,
     which applies clinically-motivated corruption (symptom recall bias,
     charting false positives, provisional-diagnosis label noise, unassessed
     symptoms). Report it as a SANITY-CHECK dataset, not a benchmark, and
     state the raw-100% fact yourself.

`audit_dataset()` measures all of this and prints a loud warning, so you can
never accidentally report a degenerate result.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

# ---------------------------------------------------------------------------
# Verified-existing Kaggle datasets (slugs checked, not guessed)
# ---------------------------------------------------------------------------
DATASETS = {
    # ~246k rows, 773 diseases, 377 one-hot symptoms. USE THIS ONE.
    "large":   "dhivyeshrk/diseases-and-symptoms-dataset",
    # 4,920 rows / 304 unique patterns, 41 diseases. Degenerate — see above.
    "classic": "kaushil268/disease-prediction-using-machine-learning",
    # 4,920 rows, same family as `classic`, plus severity/description/precaution files.
    "itachi":  "itachi9604/disease-symptom-description-dataset",
    # 25k synthetic records, 30 diseases, includes Age + Gender.
    "profile": "kundanbedmutha/healthcare-symptomsdisease-classification-dataset",
}

LABEL_CANDIDATES = ["prognosis", "diseases", "disease", "Disease", "label",
                    "Prognosis", "target", "Outcome Variable"]


# ===========================================================================
# 1. DOWNLOAD
# ===========================================================================
def ensure_kaggle_auth(verbose: bool = True) -> bool:
    """Set KAGGLE_USERNAME / KAGGLE_KEY from whichever source is available.

    Tries, in order:
      1. environment variables already set
      2. Colab Secrets  (left sidebar -> key icon -> add KAGGLE_USERNAME, KAGGLE_KEY)
      3. ~/.kaggle/kaggle.json  or  ./kaggle.json
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        if verbose:
            print("[kaggle] using credentials from environment")
        return True

    # --- Colab Secrets ---------------------------------------------------
    try:
        from google.colab import userdata  # type: ignore
        u = userdata.get("KAGGLE_USERNAME")
        k = userdata.get("KAGGLE_KEY")
        if u and k:
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = u, k
            if verbose:
                print("[kaggle] using credentials from Colab Secrets")
            return True
    except Exception:
        pass

    # --- kaggle.json -----------------------------------------------------
    import json
    for p in (os.path.expanduser("~/.kaggle/kaggle.json"), "kaggle.json",
              "/content/kaggle.json"):
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                os.environ["KAGGLE_USERNAME"] = d["username"]
                os.environ["KAGGLE_KEY"] = d["key"]
                if verbose:
                    print(f"[kaggle] using credentials from {p}")
                return True
            except Exception:
                continue

    if verbose:
        print("[kaggle] NO CREDENTIALS FOUND.\n"
              "  Get them at https://www.kaggle.com/settings -> API -> Create New Token\n"
              "  Then either:\n"
              "    (Colab)  sidebar key icon -> add KAGGLE_USERNAME and KAGGLE_KEY\n"
              "    (local)  put kaggle.json in ~/.kaggle/\n"
              "    (either) os.environ['KAGGLE_USERNAME'] = ...; os.environ['KAGGLE_KEY'] = ...")
    return False


def download(slug: str, verbose: bool = True) -> str:
    """Download a Kaggle dataset and return the local directory."""
    ensure_kaggle_auth(verbose)
    try:
        import kagglehub
    except ImportError:
        raise ImportError("pip install kagglehub") from None

    if verbose:
        print(f"[kaggle] downloading {slug} ...")
    path = kagglehub.dataset_download(slug)
    if verbose:
        print(f"[kaggle] -> {path}")
        for f in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True))[:20]:
            if os.path.isfile(f):
                print(f"          {os.path.relpath(f, path)}  "
                      f"({os.path.getsize(f)/1e6:.1f} MB)")
    return path


def find_main_csv(path: str) -> str:
    """Pick the training CSV from a downloaded dataset directory.

    Filenames differ between datasets and have changed over time, so this
    matches by intent (prefers 'train') and falls back to the largest CSV
    rather than hard-coding a name that may not exist.
    """
    csvs = [f for f in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)]
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {path}")
    train = [f for f in csvs if "train" in os.path.basename(f).lower()]
    pool = train or csvs
    return max(pool, key=os.path.getsize)


# ===========================================================================
# 2. LOAD + SCHEMA DETECTION
# ===========================================================================
def load_symptom_csv(csv_path: str, top_k_classes: int | None = None,
                     min_per_class: int = 30, verbose: bool = True
                     ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """CSV -> (binary symptom DataFrame, label Series, symptom column names).

    Handles both layouts seen in the wild:
      * WIDE  — one 0/1 column per symptom  (kaushil268, dhivyeshrk)
      * LIST  — a single text column of comma-separated symptoms (profile-style)
    """
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [c.strip() for c in df.columns]

    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:  # fall back: the last non-numeric column
        obj = [c for c in df.columns if df[c].dtype == object]
        if not obj:
            raise ValueError(f"Cannot identify a label column in {list(df.columns)[:12]}")
        label_col = obj[-1]
    if verbose:
        print(f"[kaggle] label column: '{label_col}'")

    # --- LIST layout ------------------------------------------------------
    text_sym = next((c for c in ("Symptoms", "symptoms", "text") if c in df.columns), None)
    if text_sym is not None and df[text_sym].dtype == object:
        if verbose:
            print(f"[kaggle] comma-separated symptom text detected in '{text_sym}' "
                  f"-> one-hot encoding")
        lists = df[text_sym].fillna("").astype(str).str.lower().str.split(",")
        lists = lists.apply(lambda xs: [x.strip().replace(" ", "_") for x in xs if x.strip()])
        vocab = sorted({s for xs in lists for s in xs})
        mat = np.zeros((len(df), len(vocab)), dtype=np.float32)
        idx = {s: i for i, s in enumerate(vocab)}
        for r, xs in enumerate(lists):
            for s in xs:
                mat[r, idx[s]] = 1.0
        X = pd.DataFrame(mat, columns=vocab)
        for extra in ("Age", "age", "Gender", "gender"):
            if extra in df.columns:
                v = df[extra]
                X[extra.lower()] = (pd.to_numeric(v, errors="coerce").fillna(40) / 90.0
                                    if extra.lower() == "age"
                                    else (v.astype(str).str.lower().str[0] == "m").astype(float))
        y = df[label_col].astype(str)

    # --- WIDE layout ------------------------------------------------------
    else:
        sym_cols = [c for c in df.columns if c != label_col
                    and pd.api.types.is_numeric_dtype(df[c])]
        if len(sym_cols) < 5:
            raise ValueError(f"Only {len(sym_cols)} numeric symptom columns found — "
                             f"is '{csv_path}' the right file?")
        X = df[sym_cols].fillna(0).astype(np.float32)
        X = X.clip(0, 1)          # guard against stray non-binary values
        y = df[label_col].astype(str)
        if verbose:
            print(f"[kaggle] wide binary layout: {len(sym_cols)} symptom columns")

    # --- class selection --------------------------------------------------
    counts = y.value_counts()
    keep = counts[counts >= min_per_class].index
    if top_k_classes:
        keep = counts.loc[keep].head(top_k_classes).index
    mask = y.isin(keep)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    if verbose:
        print(f"[kaggle] kept {y.nunique()} classes, {len(y)} rows "
              f"(min {min_per_class}/class)")
    return X, y, list(X.columns)


# ===========================================================================
# 3. DEGENERACY AUDIT  — run this before you trust any number
# ===========================================================================
def audit_dataset(X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> dict:
    """Measure whether the dataset is actually learnable or just a lookup table."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    n = len(X)
    uniq = X.drop_duplicates().shape[0]
    pairs = pd.concat([X, y.rename("__y__")], axis=1).drop_duplicates()
    ambiguous = int((pairs.groupby(list(X.columns))["__y__"].nunique() > 1).sum())

    yc = pd.factorize(y)[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xtr, Xte, ytr, yte = train_test_split(
                X.values, yc, test_size=0.3, random_state=0, stratify=yc)
            lr = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xtr, ytr)
            baseline = float(lr.score(Xte, yte))
    except Exception:
        baseline = float("nan")

    rep = {"rows": n, "unique_patterns": uniq,
           "duplicate_ratio": round(1 - uniq / n, 4),
           "n_classes": int(y.nunique()),
           "ambiguous_patterns": ambiguous,
           "logreg_accuracy": round(baseline, 4)}

    if verbose:
        print("\n" + "=" * 74)
        print("  DATASET AUDIT")
        print("=" * 74)
        for k, v in rep.items():
            print(f"   {k:<22} {v}")
        if baseline > 0.98 or uniq < 0.05 * n:
            print("\n   *** WARNING — THIS DATASET IS DEGENERATE ***")
            print("   A trivial linear model already scores "
                  f"{baseline:.4f} and {uniq} unique patterns cover {n} rows.")
            print("   Federated, centralized and isolated models will ALL score ~1.00,")
            print("   so the experiment cannot distinguish them and proves nothing.")
            print("   FIX: use DATASETS['large'], or call inject_ehr_realism() and")
            print("        report this as a sanity check, not a benchmark.")
        elif baseline > 0.93:
            print("\n   NOTE: baseline is high; consider more classes or more realism.")
        else:
            print("\n   OK — the task is non-trivial, results will be meaningful.")
        print("=" * 74)
    return rep


# ===========================================================================
# 4. EHR REALISM INJECTION
# ===========================================================================
def inject_ehr_realism(X: np.ndarray, y: np.ndarray, *,
                       symptom_dropout: float = 0.40,
                       symptom_false_pos: float = 0.05,
                       label_noise: float = 0.05,
                       unassessed_rate: float = 0.10,
                       n_classes: int | None = None,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a clean rule table into something that looks like real charting.

    Each corruption is clinically motivated, not arbitrary:

      symptom_dropout   patient forgets / clinician does not chart a symptom
                        that was genuinely present (recall + documentation bias).
                        THIS is what breaks the lookup-table property.
      symptom_false_pos a symptom is charted that was not really there.
      label_noise       provisional diagnosis later overturned — precisely what
                        the 14-day quarantine buffer exists to catch.
      unassessed_rate   symptom never assessed -> becomes an explicit
                        "unknown" flag rather than a silent zero.

    CALIBRATION (measured on kaushil268, 12 classes, logistic-regression
    accuracy on clean labels — pick a setting, do not guess):

        dropout  false_pos  unassessed   unique patterns   LR accuracy
          0.00     0.00       0.00              80            1.0000   <- degenerate
          0.15     0.01       0.03           1,355            0.9931
          0.25     0.02       0.05           1,435            0.9700
          0.40     0.05       0.10           1,440            0.7968   <- DEFAULT
          0.55     0.08       0.15           1,440            0.5427   <- too hard

    The defaults land near 0.80, which is a realistic clinical ceiling and
    leaves enough headroom for federated / isolated / DP differences to be
    visible. Report the setting you used — it is a hyper-parameter of the
    experiment, not a hidden detail.

    Returns (X_with_flags, y_noisy, was_unassessed_mask).
    """
    rng = np.random.default_rng(seed)
    n_classes = n_classes or int(y.max()) + 1
    Xc = X.copy().astype(np.float32)

    present = Xc > 0.5
    Xc[present & (rng.random(Xc.shape) < symptom_dropout)] = 0.0
    Xc[~present & (rng.random(Xc.shape) < symptom_false_pos)] = 1.0

    unassessed = rng.random(Xc.shape) < unassessed_rate
    Xc[unassessed] = 0.0

    yc = y.copy()
    flip = rng.random(len(yc)) < label_noise
    yc[flip] = rng.integers(0, n_classes, size=int(flip.sum()))

    # One aggregate "how much of this chart is missing" feature; a per-column
    # flag would double the input width for very little signal.
    frac_unassessed = unassessed.mean(axis=1, keepdims=True).astype(np.float32)
    return np.hstack([Xc, frac_unassessed]), yc, unassessed


# ===========================================================================
# 5. BUILD THE 3-HOSPITAL FEDERATED SPLIT
# ===========================================================================
def build_kaggle_clients(which: str = "large", *,
                         csv_path: str | None = None,
                         top_k_classes: int = 12,
                         alpha: float = 0.4,
                         test_fraction: float = 0.25,
                         realism: bool = True,
                         max_rows: int | None = 60000,
                         seed: int = cfg.SEED,
                         verbose: bool = True) -> dict:
    """Download -> audit -> partition non-IID across 3 hospitals.

    `alpha` is the Dirichlet concentration: 0.1 = extreme skew, 100 = IID.
    The synthetic pipeline gets its skew from geography; a Kaggle dataset has
    no geography, so Dirichlet is how you reproduce institutional variance.
    """
    if csv_path is None:
        slug = DATASETS.get(which, which)
        csv_path = find_main_csv(download(slug, verbose))

    X_df, y_s, sym_cols = load_symptom_csv(csv_path, top_k_classes=top_k_classes,
                                           verbose=verbose)
    audit = audit_dataset(X_df, y_s, verbose=verbose)

    if max_rows and len(X_df) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_df), max_rows, replace=False)
        X_df, y_s = X_df.iloc[idx].reset_index(drop=True), y_s.iloc[idx].reset_index(drop=True)
        if verbose:
            print(f"[kaggle] subsampled to {max_rows} rows for laptop-scale training")

    y_codes, classes = pd.factorize(y_s)
    X = X_df.values.astype(np.float32)
    y = y_codes.astype(np.int64)
    n_classes = len(classes)

    # --- hold out the global evaluation set FIRST (never corrupted) -------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_test = int(test_fraction * len(y))
    te_idx, tr_idx = perm[:n_test], perm[n_test:]

    Xte, yte = X[te_idx], y[te_idx]
    if realism:  # test set gets the same charting noise, but TRUE labels
        Xte, _, _ = inject_ehr_realism(Xte, yte, label_noise=0.0,
                                       n_classes=n_classes, seed=seed + 99)

    # --- Dirichlet non-IID partition over the training pool ---------------
    from src.data_utils import dirichlet_partition
    parts = dirichlet_partition(y[tr_idx], num_clients=len(cfg.CLIENT_IDS),
                                alpha=alpha, seed=seed)

    clients = {}
    for i, cid in enumerate(cfg.CLIENT_IDS):
        sel = tr_idx[parts[i]]
        Xi, yi = X[sel], y[sel]
        if realism:
            spec = cfg.HOSPITALS[cid]
            Xi, yi, _ = inject_ehr_realism(
                Xi, yi, label_noise=spec["label_noise"],
                unassessed_rate=spec["missing_rate"],
                n_classes=n_classes, seed=seed + i)
        clients[cid] = {"X": Xi, "y": yi, "n": len(yi)}

    in_dim = clients[cfg.CLIENT_IDS[0]]["X"].shape[1]
    if verbose:
        print(f"\n[kaggle] federated split (Dirichlet alpha={alpha})")
        for cid, c in clients.items():
            print(f"   {cid:<22} {c['n']:>6} cases")
        print(f"   {'GLOBAL HOLD-OUT':<22} {len(yte):>6} cases")
        print(f"   input dim {in_dim}, {n_classes} classes")

    return {"clients": clients, "X_test": Xte, "y_test": yte,
            "classes": list(classes), "n_classes": n_classes,
            "in_dim": in_dim, "audit": audit, "csv_path": csv_path,
            "feature_names": sym_cols + (["frac_unassessed"] if realism else [])}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect a Kaggle symptom dataset")
    ap.add_argument("--which", default="large", help=f"one of {list(DATASETS)} or a slug")
    ap.add_argument("--csv", default=None, help="skip download, use a local CSV")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()

    build_kaggle_clients(args.which, csv_path=args.csv,
                         top_k_classes=args.top_k, alpha=args.alpha)
