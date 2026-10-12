"""
src/metrics.py
==============
Every figure and table the manuscript needs. All plots are saved to
`outputs/` as 300-dpi PNGs, which is the usual Scopus/IEEE requirement.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")           # headless-safe (works over SSH and in Colab)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

import config as cfg

DPI = 300


def _save(fig, name: str) -> str:
    path = os.path.join(cfg.OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   [figure] {path}")
    return path


# ---------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, title: str, fname: str, normalize: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=range(cfg.NUM_CLASSES))
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=cfg.DISEASES)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d",
              colorbar=True, xticks_rotation=45)
    ax.set_title(title)
    return _save(fig, fname)


def plot_roc(y_true, y_prob, title: str, fname: str):
    """One-vs-rest ROC for all 8 classes + macro/micro AUC."""
    Y = label_binarize(y_true, classes=list(range(cfg.NUM_CLASSES)))
    present = [c for c in range(cfg.NUM_CLASSES) if Y[:, c].sum() > 0]

    fig, ax = plt.subplots(figsize=(8, 7))
    aucs = {}
    for c in present:
        fpr, tpr, _ = roc_curve(Y[:, c], y_prob[:, c])
        a = roc_auc_score(Y[:, c], y_prob[:, c])
        aucs[cfg.DISEASES[c]] = float(a)
        ax.plot(fpr, tpr, lw=1.6, label=f"{cfg.DISEASES[c]} (AUC={a:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, fname)

    macro = float(np.mean(list(aucs.values()))) if aucs else 0.0
    return {"per_class_auc": aucs, "macro_auc": macro}


def plot_history(history: pd.DataFrame, fname: str = "fl_convergence.png"):
    """Global accuracy / macro-F1 / loss vs communication round."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    axes[0].plot(history["round"], history["global_accuracy"], "o-", lw=2, ms=4)
    axes[0].set_title("Global accuracy vs round"); axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Accuracy")

    axes[1].plot(history["round"], history["global_macro_f1"], "s-", lw=2, ms=4, color="tab:green")
    axes[1].set_title("Global macro-F1 vs round"); axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Macro F1")

    axes[2].plot(history["round"], history["global_loss"], "^-", lw=2, ms=4, color="tab:red")
    axes[2].set_title("Global loss vs round"); axes[2].set_xlabel("Communication round")
    axes[2].set_ylabel("Cross-entropy loss")

    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, fname)


def plot_client_vs_global(history: pd.DataFrame, fname: str = "client_vs_global.png"):
    """The money figure: every hospital's LOCAL-ONLY ceiling vs the federated
    model. This is the plot that proves collaboration paid off."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["round"], history["global_accuracy"], "k-o", lw=2.5, ms=4,
            label="Federated global model")
    for cid in cfg.CLIENT_IDS:
        col = f"{cid}_local_acc"
        if col in history.columns:
            ax.plot(history["round"], history[col], "--", lw=1.4, alpha=0.85,
                    label=f"{cid} (local only)")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Accuracy on neutral multi-region test set")
    ax.set_title("Federated model vs isolated hospital models")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, fname)


def plot_label_distribution(table: pd.DataFrame, fname: str = "noniid_distribution.png"):
    """Stacked bar of each hospital's class mix — evidence of Non-IID skew."""
    data = table.drop(columns=["TOTAL"], errors="ignore")
    fig, ax = plt.subplots(figsize=(11, 5))
    data.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.6)
    ax.set_ylabel("Verified training cases")
    ax.set_title("Non-IID label distribution across hospital clients")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _save(fig, fname)


def plot_privacy_utility(sweep: pd.DataFrame, fname: str = "privacy_utility.png"):
    """Accuracy vs epsilon — the trade-off curve reviewers always ask for."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sweep["sigma"], sweep["accuracy"], "o-", color="tab:blue", lw=2)
    ax1.set_xlabel(r"Noise multiplier $\sigma$")
    ax1.set_ylabel("Global accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sweep["sigma"], sweep["epsilon"], "s--", color="tab:red", lw=2)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"Privacy budget $\epsilon$ (log scale, lower = more private)",
                   color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("Privacy-utility trade-off")
    fig.tight_layout()
    return _save(fig, fname)


# ---------------------------------------------------------------------------
def save_classification_report(y_true, y_pred, fname="classification_report.txt") -> str:
    txt = classification_report(y_true, y_pred, labels=range(cfg.NUM_CLASSES),
                                target_names=cfg.DISEASES, zero_division=0, digits=4)
    path = os.path.join(cfg.OUT_DIR, fname)
    with open(path, "w") as f:
        f.write(txt)
    print(txt)
    return path


def save_json(obj: dict, fname: str) -> str:
    path = os.path.join(cfg.OUT_DIR, fname)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    print(f"   [json]   {path}")
    return path


def save_table(df: pd.DataFrame, fname: str) -> str:
    path = os.path.join(cfg.OUT_DIR, fname)
    df.to_csv(path)
    print(f"   [table]  {path}")
    return path
