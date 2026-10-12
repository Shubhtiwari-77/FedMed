"""
src/run_simulation.py
=====================
>>> THE MAIN SCRIPT. Run this one first. <<<

Simulates the entire 3-hospital federated system on a SINGLE machine, so it
works identically in Google Colab, VS Code, or a plain terminal — no
networking, no second laptop, no GPU required.

    python src/run_simulation.py                # full run, DP on
    python src/run_simulation.py --no-dp        # ablation: DP off
    python src/run_simulation.py --no-validation# ablation: validation layer off
    python src/run_simulation.py --rounds 10    # quick smoke test
    python src/run_simulation.py --privacy-sweep

Everything it produces (figures, tables, JSON, the trained global model)
lands in `outputs/`.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src import aggregate, data_utils, engine, metrics, privacy, validation
from src.data_generator import build_all
from src.model import build_model, count_params, get_weights, set_weights


# ---------------------------------------------------------------------------
def prepare_clients(use_validation: bool = True, verbose: bool = True):
    """Generate raw hospital logs -> run the local quality gate -> tensors."""
    raw = build_all()
    clients, val_reports = {}, {}

    for cid in cfg.CLIENT_IDS:
        df = raw[cid]
        if use_validation:
            clean, reports = validation.run_validation_pipeline(df, cid, verbose=verbose)
        else:
            # Ablation: skip the gate entirely, but still impute so the tensor
            # shapes match. This is what "no safety layer" costs you.
            clean, _, _ = validation.impute_missing(df)
            reports = [{"stage": "SKIPPED", "in": len(df), "out": len(df)}]
        X, y = data_utils.encode(clean)
        clients[cid] = {"X": X, "y": y,
                        "loader": data_utils.make_loader(X, y),
                        "n": len(y)}
        val_reports[cid] = reports

    # Neutral global hold-out — bypasses the quarantine gate because its labels
    # ARE the ground truth; it is an evaluation artefact, not training data.
    #
    # Split into VALIDATION (round selection) and TEST (the number you report).
    # Selecting the best round on the test set would be a subtle leak, and with
    # DP-SGD — where accuracy peaks mid-training then decays as noise
    # accumulates — that leak would be large enough for a reviewer to catch.
    held, _, _ = validation.impute_missing(raw["GLOBAL_TEST"])
    Xh, yh = data_utils.encode(held)
    rng = np.random.default_rng(cfg.SEED)
    perm = rng.permutation(len(yh))
    cut = int(cfg.VAL_FRACTION * len(perm))
    vi, ti = perm[:cut], perm[cut:]

    val_loader = data_utils.make_loader(Xh[vi], yh[vi], shuffle=False)
    test_loader = data_utils.make_loader(Xh[ti], yh[ti], shuffle=False)

    if verbose:
        print("\n[DATA] verified training cases per hospital")
        for cid in cfg.CLIENT_IDS:
            print(f"   {cid:<22} {clients[cid]['n']:>6}")
        print(f"   {'HOLD-OUT validation':<22} {len(vi):>6}   (round selection)")
        print(f"   {'HOLD-OUT test':<22} {len(ti):>6}   (reported metrics)")

    return clients, test_loader, val_reports, val_loader


# ---------------------------------------------------------------------------
def prepare_clients_kaggle(which="large", csv_path=None, top_k_classes=12,
                           alpha=0.4, realism=True, verbose=True):
    """Same contract as `prepare_clients`, but backed by a real Kaggle dataset.

    IMPORTANT DIFFERENCE, state it in your report: a Kaggle symptom table has
    no admission dates and no clinician confidence scores, so the Phase-1
    quarantine gate and the Phase-2 schema / physiological-bounds checks have
    nothing to act on and are SKIPPED. Only the Phase-2d loss-based label
    filter runs. Fabricating admission metadata to make the other stages look
    busy would be inventing clinical data, which is worse than a missing row
    in your results table.
    """
    from src import kaggle_data

    bundle = kaggle_data.build_kaggle_clients(
        which, csv_path=csv_path, top_k_classes=top_k_classes,
        alpha=alpha, realism=realism, verbose=verbose)

    # Kaggle labels replace the synthetic ontology for the rest of the run.
    cfg.NUM_CLASSES = bundle["n_classes"]
    cfg.DISEASES = [str(c)[:22] for c in bundle["classes"]]

    clients = {}
    for cid, c in bundle["clients"].items():
        clients[cid] = {"X": c["X"], "y": c["y"], "n": c["n"],
                        "loader": data_utils.make_loader(c["X"], c["y"])}

    Xh, yh = bundle["X_test"], bundle["y_test"]
    rng = np.random.default_rng(cfg.SEED)
    perm = rng.permutation(len(yh))
    cut = int(cfg.VAL_FRACTION * len(perm))
    val_loader = data_utils.make_loader(Xh[perm[:cut]], yh[perm[:cut]], shuffle=False)
    test_loader = data_utils.make_loader(Xh[perm[cut:]], yh[perm[cut:]], shuffle=False)

    # --- DP needs volume. Warn loudly rather than let the user publish 8%. ---
    smallest = min(c["n"] for c in clients.values())
    if smallest < 3000:
        print("\n" + "!" * 74)
        print(f"  WARNING: smallest hospital has only {smallest} cases.")
        print("  DP-SGD needs thousands of records per client. Measured on the")
        print("  'classic' dataset (~1,000 cases/hospital):")
        print("       no DP  -> 0.766 accuracy      DP sigma=1.5 -> 0.270")
        print("       no DP, 41 classes -> 0.637    DP, 41 classes -> 0.087")
        print("  The pipeline is fine; there is simply not enough data to")
        print("  average the noise away. Either use --kaggle-which large")
        print("  (246k rows), or run --no-dp and say so explicitly.")
        print("!" * 74)

    reports = {cid: [{"stage": "kaggle_source", "in": c["n"], "out": c["n"]}]
               for cid, c in bundle["clients"].items()}
    return clients, test_loader, reports, val_loader, bundle["in_dim"], bundle


# ---------------------------------------------------------------------------
def run_federated(clients, test_loader, *, val_loader=None, in_dim=None,
                  rounds=cfg.NUM_ROUNDS,
                  local_epochs=cfg.LOCAL_EPOCHS, dp_mode=cfg.DP_MODE,
                  noise_multiplier=cfg.DP_NOISE_MULTIPLIER,
                  clip_norm=cfg.DP_CLIP_NORM, lr=None, batch_size=None,
                  use_loss_filter=True,
                  track_local_baselines=True, verbose=True, seed=cfg.SEED):
    """The FedAvg loop.

    dp_mode:
      "record" -> DP-SGD inside local training   (default; protects a patient)
      "client" -> clip+noise the sent update     (ablation; protects a hospital)
      "none"   -> no DP                          (utility-ceiling baseline)

    Returns (global_weights, history_df, final_eval, global_model, local_only).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    # DP-SGD clips every gradient to norm C, which shrinks the effective step;
    # it therefore needs a larger learning rate than plain local training.
    if lr is None:
        lr = cfg.LEARNING_RATE_DP if dp_mode == "record" else cfg.LEARNING_RATE
    # A smaller DP batch means a smaller sampling rate q, which buys a large
    # amount of privacy amplification for very little utility.
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE_DP if dp_mode == "record" else cfg.BATCH_SIZE

    in_dim = in_dim if in_dim is not None else data_utils.NUM_FEATURES
    global_model = build_model(in_dim, seed)
    global_w = get_weights(global_model)

    # Isolated per-hospital models = the "no collaboration" control group.
    local_only = {}
    if track_local_baselines:
        for cid in clients:
            local_only[cid] = build_model(in_dim, seed)

    sel_loader = val_loader if val_loader is not None else test_loader
    best = {"score": -1.0, "round": 0, "weights": global_w}

    class_w = {cid: engine.balanced_class_weights(clients[cid]["y"]) for cid in clients}
    dp_steps = {cid: 0 for cid in clients}
    sample_rates = {cid: 0.0 for cid in clients}
    history = []

    if verbose:
        print(f"\n[MODEL] DiagnosisNet  in_dim={in_dim}  params={count_params(global_model):,}")
        dp_desc = ("OFF" if dp_mode == "none"
                   else f"{dp_mode.upper()}-level (sigma={noise_multiplier:.2f}, C={clip_norm:.2f})")
        print(f"[FL]    rounds={rounds}  local_epochs={local_epochs}  DP={dp_desc}")
        print("\n" + "=" * 96)
        print(f"{'Rnd':>4} {'GlobalAcc':>10} {'GlobalF1':>9} {'GlobalLoss':>11} "
              f"{'AvgLocalTrain':>14} {'Filtered':>9} {'||dW||':>9} {'sec':>6}")
        print("=" * 96)

    for rnd in range(1, rounds + 1):
        t0 = time.time()
        client_weights, client_sizes, train_accs = [], [], []
        total_filtered = 0

        for cid, c in clients.items():
            X, y = c["X"], c["y"]

            # -------- Phase 2d: loss-based label filter -------------------
            keep = np.ones(len(y), dtype=bool)
            if use_loss_filter and rnd >= cfg.LOSS_FILTER_START_ROUND:
                probe = build_model(in_dim, seed)
                set_weights(probe, global_w)
                losses = engine.per_sample_losses(probe, X, y)
                keep = validation.loss_based_mask(losses)
                total_filtered += int((~keep).sum())

            Xk, yk = X[keep], y[keep]
            loader = data_utils.make_loader(Xk, yk)

            # -------- Phase 3: local training on hospital hardware --------
            local_model = build_model(in_dim, seed)
            set_weights(local_model, global_w)

            if dp_mode == "record":
                # Record-level DP: noise goes in DURING local optimisation
                stats = engine.local_train_dp(
                    local_model, Xk, yk, epochs=local_epochs, lr=lr,
                    clip_norm=clip_norm, noise_multiplier=noise_multiplier,
                    class_weights=class_w[cid], batch_size=batch_size)
                dp_steps[cid] += stats["dp_steps"]
                # q can drift as the loss filter shrinks the local set; keep the
                # LARGEST q seen, since that is the worst case for privacy.
                sample_rates[cid] = max(sample_rates[cid], stats["sample_rate"])
            else:
                stats = engine.local_train(local_model, loader, epochs=local_epochs,
                                           lr=lr, class_weights=class_w[cid])

            local_w = get_weights(local_model)

            # -------- Client-level DP on the transmitted update -----------
            if dp_mode == "client":
                local_w, _ = privacy.privatise(global_w, local_w, clip_norm,
                                               noise_multiplier,
                                               num_clients=len(clients), rng=rng)

            # -------- Phase 3b: transmit weights ONLY ---------------------
            client_weights.append(local_w)
            client_sizes.append(int(keep.sum()))
            train_accs.append(stats["train_acc"])

            # -------- control group: isolated hospital model ---------------
            # Trained with the SAME privacy mechanism as the federated model,
            # otherwise the comparison is apples-to-oranges: a non-private
            # local model would beat a private federated one for reasons that
            # have nothing to do with federation.
            if track_local_baselines:
                if dp_mode == "record":
                    engine.local_train_dp(
                        local_only[cid], Xk, yk, epochs=local_epochs, lr=lr,
                        clip_norm=clip_norm, noise_multiplier=noise_multiplier,
                        class_weights=class_w[cid], batch_size=batch_size)
                else:
                    engine.local_train(local_only[cid], loader, epochs=local_epochs,
                                       lr=lr, class_weights=class_w[cid])

        # -------- Phase 4: central server FedAvg --------------------------
        prev_w = global_w
        global_w = aggregate.fedavg(client_weights, client_sizes)
        dw = aggregate.weight_delta_norm(global_w, prev_w)

        # -------- evaluate the redistributed global model -----------------
        set_weights(global_model, global_w)
        sv = engine.evaluate(global_model, sel_loader)
        if sv["macro_f1"] > best["score"]:
            best = {"score": sv["macro_f1"], "round": rnd,
                    "weights": [w.copy() for w in global_w]}
        ev = engine.evaluate(global_model, test_loader)

        row = {"round": rnd, "val_macro_f1": sv["macro_f1"],
               "global_accuracy": ev["accuracy"],
               "global_macro_f1": ev["macro_f1"], "global_loss": ev["loss"],
               "avg_local_train_acc": float(np.mean(train_accs)),
               "filtered_samples": total_filtered, "weight_delta": dw}
        if track_local_baselines:
            for cid, m in local_only.items():
                row[f"{cid}_local_acc"] = engine.evaluate(m, test_loader)["accuracy"]
        history.append(row)

        if verbose:
            print(f"{rnd:>4} {ev['accuracy']:>10.4f} {ev['macro_f1']:>9.4f} "
                  f"{ev['loss']:>11.4f} {np.mean(train_accs):>14.4f} "
                  f"{total_filtered:>9} {dw:>9.4f} {time.time()-t0:>6.1f}")

    # Restore the round chosen on VALIDATION data, then report on TEST data.
    global_w = best["weights"]
    set_weights(global_model, global_w)
    final = engine.evaluate(global_model, test_loader)
    final["dp_steps"] = dp_steps
    final["sample_rates"] = sample_rates
    final["per_client_dp"] = {c: (dp_steps[c], sample_rates[c]) for c in clients}
    final["selected_round"] = best["round"]
    if verbose:
        print(f"\n[SELECT] best round on validation data: {best['round']}/{rounds} "
              f"(val macro-F1 {best['score']:.4f}) -> reporting that model on the test set")
    return global_w, pd.DataFrame(history), final, global_model, local_only


# ---------------------------------------------------------------------------
def privacy_sweep(clients, test_loader, val_loader=None, in_dim=None, sigmas=(0.0, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0),
                  rounds=12, dp_mode="record"):
    """Accuracy and epsilon at several noise levels -> Table 4 of the paper."""
    rows = []
    for s in sigmas:
        mode = "none" if s <= 0 else dp_mode
        print(f"\n--- privacy sweep: sigma = {s}  (mode={mode}) ---")
        _, _, final, _, _ = run_federated(
            clients, test_loader, val_loader=val_loader, in_dim=in_dim,
            rounds=rounds, dp_mode=mode,
            noise_multiplier=s, track_local_baselines=False, verbose=False)

        if s <= 0:
            eps = float("nan")
        elif dp_mode == "record":
            eps, _ = privacy.epsilon_worst_client(final["per_client_dp"], s)
        else:
            eps = privacy.epsilon_client_level(rounds, s)

        rows.append({"sigma": s, "epsilon": eps, "accuracy": final["accuracy"],
                     "macro_f1": final["macro_f1"]})
        print(f"    acc={final['accuracy']:.4f}  macroF1={final['macro_f1']:.4f}  eps={eps:.2f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="FedMed-FL simulation")
    ap.add_argument("--data", choices=["synthetic", "kaggle"], default="synthetic",
                    help="synthetic = context-aware generator (default); "
                         "kaggle = real Kaggle symptom dataset")
    ap.add_argument("--kaggle-which", default="large",
                    help="large | classic | itachi | profile | any owner/slug")
    ap.add_argument("--kaggle-csv", default=None,
                    help="skip the download and use a local CSV")
    ap.add_argument("--top-k-classes", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Dirichlet non-IID concentration (kaggle mode)")
    ap.add_argument("--no-realism", action="store_true",
                    help="kaggle mode: skip EHR corruption (will be degenerate)")
    ap.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS)
    ap.add_argument("--local-epochs", type=int, default=cfg.LOCAL_EPOCHS)
    ap.add_argument("--dp-mode", choices=["record", "client", "none"], default=cfg.DP_MODE,
                    help="record = DP-SGD per patient (default); "
                         "client = DP on the sent update; none = no DP")
    ap.add_argument("--no-dp", action="store_true", help="shorthand for --dp-mode none")
    ap.add_argument("--no-validation", action="store_true", help="disable the Phase-2 quality gate")
    ap.add_argument("--no-loss-filter", action="store_true")
    ap.add_argument("--sigma", type=float, default=cfg.DP_NOISE_MULTIPLIER)
    ap.add_argument("--clip", type=float, default=cfg.DP_CLIP_NORM)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--privacy-sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()
    if args.no_dp:
        args.dp_mode = "none"

    print("=" * 96)
    print("  FedMed-FL : Privacy-Preserving Collaborative Medical Diagnostics")
    print("=" * 96)

    if args.data == "kaggle":
        clients, test_loader, val_reports, val_loader, in_dim, bundle = \
            prepare_clients_kaggle(args.kaggle_which, csv_path=args.kaggle_csv,
                                   top_k_classes=args.top_k_classes,
                                   alpha=args.alpha, realism=not args.no_realism)
    else:
        clients, test_loader, val_reports, val_loader = prepare_clients(
            use_validation=not args.no_validation)
        in_dim = data_utils.NUM_FEATURES

    # Table 1 — Non-IID distribution
    table = data_utils.label_distribution_table({c: clients[c]["y"] for c in clients})
    print("\n[TABLE 1] Non-IID label distribution")
    print(table.to_string())
    metrics.save_table(table, "table1_noniid_distribution.csv")
    metrics.plot_label_distribution(table)

    if args.privacy_sweep:
        sweep = privacy_sweep(clients, test_loader, val_loader=val_loader, in_dim=in_dim,
                              dp_mode=args.dp_mode if args.dp_mode != "none" else "record")
        metrics.save_table(sweep, "table4_privacy_sweep.csv")
        metrics.plot_privacy_utility(sweep)
        print("\n[TABLE 4] Privacy-utility trade-off")
        print(sweep.to_string(index=False))
        return

    gw, history, final, global_model, local_only = run_federated(
        clients, test_loader, val_loader=val_loader, in_dim=in_dim,
        rounds=args.rounds, local_epochs=args.local_epochs,
        dp_mode=args.dp_mode, noise_multiplier=args.sigma, clip_norm=args.clip,
        lr=args.lr, use_loss_filter=not args.no_loss_filter, seed=args.seed)

    print("\n" + "=" * 96)
    print(f"[FINAL] Global accuracy : {final['accuracy']:.4f}   "
          f"(round {final['selected_round']})")
    print(f"[FINAL] Global macro-F1 : {final['macro_f1']:.4f}")

    # ---- figures ----------------------------------------------------------
    print("\n[OUTPUT] writing figures and tables to outputs/")
    metrics.plot_history(history)
    metrics.plot_client_vs_global(history)
    metrics.plot_confusion(final["y_true"], final["y_pred"],
                           "Federated global model — normalised confusion matrix",
                           "confusion_matrix_federated.png")
    roc = metrics.plot_roc(final["y_true"], final["y_prob"],
                           "Federated global model — one-vs-rest ROC",
                           "roc_federated.png")
    metrics.save_classification_report(final["y_true"], final["y_pred"])
    metrics.save_table(history, "fl_history.csv")

    # ---- Table 3: federated vs isolated ------------------------------------
    dp_tag = "" if args.dp_mode == "none" else f" + {args.dp_mode}-DP"
    rows = [{"model": f"Federated global (FedAvg{dp_tag})",
             "accuracy": final["accuracy"], "macro_f1": final["macro_f1"],
             "macro_auc": roc["macro_auc"], "sees_data_of": "3 hospitals (weights only)"}]
    for cid, m in local_only.items():
        ev = engine.evaluate(m, test_loader)
        r = metrics.plot_roc(ev["y_true"], ev["y_prob"],
                             f"{cid} local-only — ROC", f"roc_{cid}.png")
        rows.append({"model": f"{cid} (isolated{dp_tag})", "accuracy": ev["accuracy"],
                     "macro_f1": ev["macro_f1"], "macro_auc": r["macro_auc"],
                     "sees_data_of": "1 hospital"})
    comp = pd.DataFrame(rows)
    print("\n[TABLE 3] Federated vs isolated hospital models")
    print(comp.to_string(index=False))
    metrics.save_table(comp, "table3_federated_vs_isolated.csv")

    # ---- privacy accounting -----------------------------------------------
    prep = privacy.privacy_report(
        args.dp_mode, per_client=final.get("per_client_dp"),
        num_rounds=args.rounds, noise_multiplier=args.sigma, clip_norm=args.clip)
    print("\n[PRIVACY]")
    for k, v in prep.items():
        print(f"   {k:<26} {v}")

    metrics.save_json({
        "final": {k: v for k, v in final.items()
                  if not isinstance(v, np.ndarray)
                  and k not in ("dp_steps", "sample_rates", "per_client_dp")},
        "roc": roc, "privacy": prep,
        "validation_reports": val_reports,
        "config": {"rounds": args.rounds, "local_epochs": args.local_epochs,
                   "dp_mode": args.dp_mode, "sigma": args.sigma, "clip": args.clip,
                   "validation": not args.no_validation},
    }, "results_summary.json")

    # ---- persist the global model for the CDSS -----------------------------
    ckpt = os.path.join(cfg.OUT_DIR, "global_model.pt")
    torch.save({"state_dict": global_model.state_dict(),
                "in_dim": in_dim, "data_source": args.data,
                "feature_names": data_utils.feature_names(),
                "diseases": cfg.DISEASES}, ckpt)
    print(f"\n[OUTPUT] global model saved -> {ckpt}")
    print("[NEXT]   run:  python src/cdss.py   (doctor-facing inference demo)")


if __name__ == "__main__":
    main()
