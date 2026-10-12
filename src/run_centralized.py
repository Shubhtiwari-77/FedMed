"""
src/run_centralized.py
======================
The UPPER-BOUND control experiment.

Pools all three hospitals' data into one dataset and trains a single model —
i.e. it does exactly what HIPAA / GDPR / DISHA forbid. The gap between this
number and the federated number is the "price of privacy", and it is the one
figure every reviewer will look for.

    python src/run_centralized.py

Compute budget is matched to the federated run (CENTRAL_EPOCHS =
NUM_ROUNDS x LOCAL_EPOCHS) so the comparison is fair.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src import data_utils, engine, metrics
from src.model import build_model
from src.run_simulation import prepare_clients, prepare_clients_kaggle


def main():
    ap = argparse.ArgumentParser(description="Centralized (data-pooling) baseline")
    ap.add_argument("--data", choices=["synthetic", "kaggle"], default="synthetic")
    ap.add_argument("--kaggle-which", default="large")
    ap.add_argument("--kaggle-csv", default=None)
    ap.add_argument("--top-k-classes", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=cfg.CENTRAL_EPOCHS)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()

    print("=" * 80)
    print("  CENTRALIZED BASELINE — pools all hospital data (privacy-violating)")
    print("=" * 80)

    if args.data == "kaggle":
        clients, test_loader, _, _, in_dim, _ = prepare_clients_kaggle(
            args.kaggle_which, csv_path=args.kaggle_csv,
            top_k_classes=args.top_k_classes, alpha=args.alpha)
    else:
        clients, test_loader, _, _ = prepare_clients(verbose=True)
        in_dim = data_utils.NUM_FEATURES

    X = np.concatenate([c["X"] for c in clients.values()])
    y = np.concatenate([c["y"] for c in clients.values()])
    print(f"\n[DATA] pooled training set: {len(y)} cases from {len(clients)} hospitals")

    torch.manual_seed(args.seed)
    model = build_model(in_dim, args.seed)
    loader = data_utils.make_loader(X, y)
    cw = engine.balanced_class_weights(y)

    rows = []
    chunk = max(1, args.epochs // 15)
    done = 0
    while done < args.epochs:
        e = min(chunk, args.epochs - done)
        engine.local_train(model, loader, epochs=e, lr=cfg.LEARNING_RATE, class_weights=cw)
        done += e
        ev = engine.evaluate(model, test_loader)
        rows.append({"epoch": done, "accuracy": ev["accuracy"],
                     "macro_f1": ev["macro_f1"], "loss": ev["loss"]})
        print(f"   epoch {done:>4}/{args.epochs}   acc={ev['accuracy']:.4f}   "
              f"macroF1={ev['macro_f1']:.4f}   loss={ev['loss']:.4f}")

    final = engine.evaluate(model, test_loader)
    print("\n" + "=" * 80)
    print(f"[FINAL] Centralized accuracy : {final['accuracy']:.4f}")
    print(f"[FINAL] Centralized macro-F1 : {final['macro_f1']:.4f}")

    metrics.plot_confusion(final["y_true"], final["y_pred"],
                           "Centralized baseline — normalised confusion matrix",
                           "confusion_matrix_centralized.png")
    roc = metrics.plot_roc(final["y_true"], final["y_prob"],
                           "Centralized baseline — one-vs-rest ROC",
                           "roc_centralized.png")
    metrics.save_table(pd.DataFrame(rows), "centralized_history.csv")
    metrics.save_json({"accuracy": final["accuracy"], "macro_f1": final["macro_f1"],
                       "macro_auc": roc["macro_auc"], "epochs": args.epochs,
                       "pooled_samples": int(len(y))}, "centralized_summary.json")

    fl_path = os.path.join(cfg.OUT_DIR, "results_summary.json")
    if os.path.exists(fl_path):
        import json
        fl = json.load(open(fl_path))["final"]
        gap = final["accuracy"] - fl["accuracy"]
        print("\n[COMPARISON]")
        print(f"   Centralized (pools raw data, ILLEGAL) : {final['accuracy']:.4f}")
        print(f"   Federated   (raw data never moves)    : {fl['accuracy']:.4f}")
        print(f"   Price of privacy                      : {gap:+.4f} accuracy")
    else:
        print("\n[NOTE] run src/run_simulation.py first to see the side-by-side gap.")


if __name__ == "__main__":
    main()
