"""
federated/server.py
===================
THE CENTRAL SERVER NODE — run this on Laptop 1 (HP Victus).

This is the REAL distributed deployment described in Section 5 of your
proposal. Unlike `src/run_simulation.py` (which fakes the network), this
actually opens a gRPC socket and waits for hospital clients to connect over
your Wi-Fi.

    # Laptop 1 — start the server FIRST
    python federated/server.py --rounds 30 --min-clients 3

The server:
  * initialises the global model and broadcasts W_t
  * waits for `--min-clients` hospitals to report in
  * runs weighted FedAvg over whatever they send back
  * evaluates the new global model on the neutral test set
  * redistributes W_{t+1} and repeats

It NEVER receives a patient record — only float tensors.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src import data_utils, engine, metrics, validation
from src.data_generator import build_all
from src.model import build_model, count_params, set_weights


# ---------------------------------------------------------------------------
def build_test_loader():
    """The neutral, multi-region hold-out set the server evaluates on.

    In a real hospital consortium this would be a small IRB-approved,
    de-identified benchmark set — the only data the orchestrator is allowed
    to hold. Here it is generated locally.
    """
    raw = build_all()["GLOBAL_TEST"]
    clean, _, _ = validation.impute_missing(raw)
    X, y = data_utils.encode(clean)
    return data_utils.make_loader(X, y, shuffle=False)


def weighted_average(results: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate client-reported metrics by sample count — same rule as FedAvg."""
    total = sum(n for n, _ in results) or 1
    out: Metrics = {}
    for key in ("accuracy", "loss", "macro_f1"):
        vals = [(n, m[key]) for n, m in results if key in m]
        if vals:
            out[key] = sum(n * v for n, v in vals) / sum(n for n, _ in vals)
    out["total_examples"] = total
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="FedMed-FL Flower server")
    ap.add_argument("--address", default=cfg.SERVER_ADDRESS,
                    help="bind address, e.g. 0.0.0.0:8080")
    ap.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS)
    ap.add_argument("--min-clients", type=int, default=cfg.NUM_CLIENTS)
    ap.add_argument("--local-epochs", type=int, default=cfg.LOCAL_EPOCHS)
    ap.add_argument("--dp-mode", choices=["record", "client", "none"], default=cfg.DP_MODE)
    ap.add_argument("--sigma", type=float, default=cfg.DP_NOISE_MULTIPLIER)
    args = ap.parse_args()

    torch.manual_seed(cfg.SEED)
    in_dim = data_utils.NUM_FEATURES
    global_model = build_model(in_dim, cfg.SEED)
    test_loader = build_test_loader()
    history_rows: list[dict] = []

    print("=" * 84)
    print("  FedMed-FL  |  CENTRAL SERVER NODE")
    print("=" * 84)
    print(f"  listening on      : {args.address}")
    print(f"  rounds            : {args.rounds}")
    print(f"  waiting for       : {args.min_clients} hospital client(s)")
    print(f"  model             : DiagnosisNet  in_dim={in_dim}  "
          f"params={count_params(global_model):,}")
    print(f"  DP mode           : {args.dp_mode} (sigma={args.sigma})")
    print("=" * 84)
    print("  Now start the clients. On THIS laptop use 127.0.0.1:8080;")
    print("  on the OTHER laptop use this machine's LAN IP, e.g. 192.168.1.7:8080")
    print("  (find it with `ipconfig` on Windows or `ifconfig | grep inet` on macOS)")
    print("=" * 84 + "\n")

    # --- server-side evaluation of the aggregated global model -------------
    def evaluate_fn(server_round: int, parameters, config):
        set_weights(global_model, parameters)
        ev = engine.evaluate(global_model, test_loader)
        history_rows.append({"round": server_round,
                             "global_accuracy": ev["accuracy"],
                             "global_macro_f1": ev["macro_f1"],
                             "global_loss": ev["loss"]})
        print(f"  [SERVER] round {server_round:>3}  "
              f"acc={ev['accuracy']:.4f}  macroF1={ev['macro_f1']:.4f}  "
              f"loss={ev['loss']:.4f}")
        return ev["loss"], {"accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"]}

    # --- config pushed down to every client each round ---------------------
    def fit_config(server_round: int) -> dict:
        return {"server_round": server_round,
                "local_epochs": args.local_epochs,
                "dp_mode": args.dp_mode,
                "sigma": args.sigma,
                "clip_norm": cfg.DP_CLIP_NORM}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=ndarrays_to_parameters(
            [v.cpu().numpy() for v in global_model.state_dict().values()]),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    hist = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # --- persist everything ------------------------------------------------
    print("\n" + "=" * 84)
    print("  TRAINING COMPLETE")
    print("=" * 84)

    if history_rows:
        df = pd.DataFrame(history_rows)
        metrics.save_table(df, "flower_server_history.csv")
        metrics.plot_history(df, "flower_convergence.png")
        best = df.loc[df["global_accuracy"].idxmax()]
        print(f"  best accuracy : {best['global_accuracy']:.4f} at round {int(best['round'])}")

    ev = engine.evaluate(global_model, test_loader)
    metrics.plot_confusion(ev["y_true"], ev["y_pred"],
                           "Flower deployment — global model confusion matrix",
                           "confusion_matrix_flower.png")
    metrics.plot_roc(ev["y_true"], ev["y_prob"],
                     "Flower deployment — one-vs-rest ROC", "roc_flower.png")

    ckpt = os.path.join(cfg.OUT_DIR, "global_model.pt")
    torch.save({"state_dict": global_model.state_dict(), "in_dim": in_dim,
                "feature_names": data_utils.feature_names(),
                "diseases": cfg.DISEASES}, ckpt)
    print(f"  global model saved -> {ckpt}")
    print(f"  next: python src/cdss.py")


if __name__ == "__main__":
    main()
