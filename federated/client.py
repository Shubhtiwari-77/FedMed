"""
federated/client.py
===================
A HOSPITAL NODE. Run one instance per hospital.

    # Laptop 1 (same machine as the server)
    python federated/client.py --client-id H1_Max_Delhi   --server 127.0.0.1:8080

    # Laptop 2 — replace with the SERVER laptop's LAN IP
    python federated/client.py --client-id H2_Apollo_Chennai --server 192.168.1.7:8080
    python federated/client.py --client-id H3_SKIMS_Srinagar --server 192.168.1.7:8080

Each client:
  1. loads ONLY its own hospital's admission log
  2. runs the Phase-1 quarantine gate + Phase-2 validation locally
  3. receives global weights, trains locally for E epochs
  4. applies differential privacy
  5. returns WEIGHTS ONLY

Nothing else crosses the process boundary. That is the whole point.
"""

from __future__ import annotations

import argparse
import os
import sys

import flwr as fl
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src import data_utils, engine, privacy, validation
from src.data_generator import build_hospital_dataframe
from src.model import build_model, get_weights, set_weights


# ---------------------------------------------------------------------------
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, seed: int = cfg.SEED):
        self.client_id = client_id
        spec = cfg.HOSPITALS[client_id]
        idx = cfg.CLIENT_IDS.index(client_id)

        # ---- Phase 1: this hospital's own raw admission log --------------
        raw = build_hospital_dataframe(
            n_samples=spec["n_samples"], region_mix=spec["region_mix"],
            month_mix=spec["month_mix"], label_noise=spec["label_noise"],
            missing_rate=spec["missing_rate"], seed=seed + idx)

        # ---- Phase 2: local quality gate (never leaves the hospital) -----
        clean, self.reports = validation.run_validation_pipeline(raw, client_id)

        self.X, self.y = data_utils.encode(clean)
        self.class_weights = engine.balanced_class_weights(self.y)

        # Local hold-out so the client can report honest eval metrics
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self.y))
        cut = int(0.85 * len(perm))
        self.tr, self.te = perm[:cut], perm[cut:]
        self.test_loader = data_utils.make_loader(
            self.X[self.te], self.y[self.te], shuffle=False)

        self.model = build_model(data_utils.NUM_FEATURES, seed)
        self.rng = np.random.default_rng(seed + 7)
        self.dp_steps = 0
        self.sample_rate = 0.0

        print(f"\n[{client_id}] ready — {len(self.tr)} verified training cases, "
              f"{len(self.te)} local hold-out")
        dist = np.bincount(self.y[self.tr], minlength=cfg.NUM_CLASSES)
        for d, n in zip(cfg.DISEASES, dist):
            print(f"    {d:<22} {n:>5}")

    # -- Flower hooks -------------------------------------------------------
    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        rnd = int(config.get("server_round", 1))
        epochs = int(config.get("local_epochs", cfg.LOCAL_EPOCHS))
        dp_mode = str(config.get("dp_mode", cfg.DP_MODE))
        sigma = float(config.get("sigma", cfg.DP_NOISE_MULTIPLIER))
        clip = float(config.get("clip_norm", cfg.DP_CLIP_NORM))

        set_weights(self.model, parameters)
        global_w = [np.array(p) for p in parameters]

        Xtr, ytr = self.X[self.tr], self.y[self.tr]

        # ---- Phase 2d: loss-based label filter --------------------------
        n_filtered = 0
        if rnd >= cfg.LOSS_FILTER_START_ROUND:
            losses = engine.per_sample_losses(self.model, Xtr, ytr)
            keep = validation.loss_based_mask(losses)
            n_filtered = int((~keep).sum())
            Xtr, ytr = Xtr[keep], ytr[keep]

        # ---- Phase 3: local training ------------------------------------
        if dp_mode == "record":
            stats = engine.local_train_dp(
                self.model, Xtr, ytr, epochs=epochs, lr=cfg.LEARNING_RATE_DP,
                clip_norm=clip, noise_multiplier=sigma,
                class_weights=self.class_weights)
            self.dp_steps += stats["dp_steps"]
            self.sample_rate = stats["sample_rate"]
        else:
            loader = data_utils.make_loader(Xtr, ytr)
            stats = engine.local_train(self.model, loader, epochs=epochs,
                                       lr=cfg.LEARNING_RATE,
                                       class_weights=self.class_weights)

        local_w = get_weights(self.model)

        # ---- Client-level DP on the transmitted update -------------------
        if dp_mode == "client":
            local_w, raw_norm = privacy.privatise(
                global_w, local_w, clip, sigma, cfg.NUM_CLIENTS, self.rng)
        else:
            raw_norm = float(np.sqrt(sum(float(np.sum((l - g) ** 2))
                                         for l, g in zip(local_w, global_w))))

        print(f"  [{self.client_id}] round {rnd:>3}  n={len(ytr):<5} "
              f"filtered={n_filtered:<4} train_acc={stats['train_acc']:.4f}  "
              f"||dW||={raw_norm:.3f}")

        # ---- Phase 3b: WEIGHTS ONLY leave this machine -------------------
        return local_w, len(ytr), {
            "accuracy": float(stats["train_acc"]),
            "loss": float(stats["train_loss"]),
            "filtered": n_filtered,
        }

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        ev = engine.evaluate(self.model, self.test_loader)
        return float(ev["loss"]), len(self.te), {
            "accuracy": float(ev["accuracy"]),
            "macro_f1": float(ev["macro_f1"]),
            "loss": float(ev["loss"]),
        }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="FedMed-FL hospital client")
    ap.add_argument("--client-id", required=True, choices=cfg.CLIENT_IDS)
    ap.add_argument("--server", default=cfg.CLIENT_CONNECT_ADDRESS,
                    help="server address, e.g. 192.168.1.7:8080")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()

    print("=" * 84)
    print(f"  FedMed-FL  |  HOSPITAL CLIENT: {args.client_id}")
    print(f"  connecting to server at {args.server}")
    print("=" * 84)

    client = HospitalClient(args.client_id, args.seed)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
