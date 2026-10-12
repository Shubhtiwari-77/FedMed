"""
src/engine.py
=============
Local training / evaluation loops. Everything here runs INSIDE one hospital.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def local_train(model: nn.Module, loader: DataLoader,
                epochs: int = cfg.LOCAL_EPOCHS,
                lr: float = cfg.LEARNING_RATE,
                class_weights: np.ndarray | None = None) -> dict:
    """Train the local model for E epochs. Returns training stats."""
    dev = _device()
    model.to(dev).train()

    w = None
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)

    total_loss, total_n, correct = 0.0, 0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            # Gradient clipping keeps the per-round update norm well-behaved,
            # which in turn makes DP clipping bite less hard on utility.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item() * yb.size(0)
            total_n += yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()

    return {"train_loss": total_loss / max(total_n, 1),
            "train_acc": correct / max(total_n, 1),
            "num_examples": len(loader.dataset)}


def local_train_dp(model: nn.Module, X: np.ndarray, y: np.ndarray,
                   epochs: int = cfg.LOCAL_EPOCHS,
                   lr: float = cfg.LEARNING_RATE,
                   clip_norm: float = cfg.DP_CLIP_NORM,
                   noise_multiplier: float = cfg.DP_NOISE_MULTIPLIER,
                   batch_size: int = cfg.BATCH_SIZE,
                   class_weights: np.ndarray | None = None,
                   seed: int | None = None) -> dict:
    """RECORD-LEVEL DP-SGD (Abadi et al., 2016) for one hospital, one round.

    Differences from `local_train` that matter for the privacy proof:
      * POISSON subsampling — every record is included independently with
        probability q = batch_size / n. This is what the RDP accountant in
        `privacy.py` assumes; shuffled fixed-size batches would make the
        reported epsilon only an approximation.
      * PER-EXAMPLE gradient clipping to L2 norm C, so one patient's
        influence on the released update is bounded by C.
      * Gaussian noise N(0, (sigma*C)^2) added to the CLIPPED SUM, then
        divided by the EXPECTED batch size (q*n), not the realised one.
    """
    from src import privacy

    dev = _device()
    model.to(dev).train()
    if seed is not None:
        torch.manual_seed(seed)

    n = len(y)
    q = min(1.0, batch_size / max(n, 1))
    steps_per_epoch = max(1, int(np.ceil(n / batch_size)))
    expected_bs = max(q * n, 1.0)

    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    cw = None
    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=dev)

    grad_fn = privacy.make_per_sample_grad_fn(model, class_weights=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)

    total_loss, total_n, correct, n_steps = 0.0, 0, 0, 0
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            # ---- Poisson subsampling -------------------------------------
            mask = torch.rand(n, device=dev) < q
            if not bool(mask.any()):
                continue
            xb, yb = Xt[mask], yt[mask]

            params = {k: v.detach() for k, v in model.named_parameters()}
            buffers = {k: v.detach() for k, v in model.named_buffers()}

            psg = grad_fn(params, buffers, xb, yb)
            noised = privacy.clip_and_noise_per_sample(
                psg, clip_norm, noise_multiplier, expected_bs, dev)

            opt.zero_grad(set_to_none=True)
            for name, p in model.named_parameters():
                p.grad = noised[name].detach().clone()
            opt.step()
            n_steps += 1

            with torch.no_grad():
                out = model(xb)
                total_loss += nn.functional.cross_entropy(out, yb).item() * yb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                total_n += yb.size(0)

    return {"train_loss": total_loss / max(total_n, 1),
            "train_acc": correct / max(total_n, 1),
            "num_examples": n, "dp_steps": n_steps, "sample_rate": q}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Returns loss, accuracy, macro-F1, and the raw arrays for plotting."""
    from sklearn.metrics import f1_score

    dev = _device()
    model.to(dev).eval()
    criterion = nn.CrossEntropyLoss()

    losses, ys, preds, probs = 0.0, [], [], []
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        out = model(xb)
        losses += criterion(out, yb).item() * yb.size(0)
        n += yb.size(0)
        p = torch.softmax(out, dim=1)
        ys.append(yb.cpu().numpy())
        preds.append(out.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.zeros(0, dtype=int)
    y_pred = np.concatenate(preds) if preds else np.zeros(0, dtype=int)
    y_prob = np.concatenate(probs) if probs else np.zeros((0, cfg.NUM_CLASSES))

    return {
        "loss": losses / max(n, 1),
        "accuracy": float((y_true == y_pred).mean()) if n else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if n else 0.0,
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
        "num_examples": n,
    }


@torch.no_grad()
def per_sample_losses(model: nn.Module, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Un-reduced CE loss for every local sample — feeds the label filter."""
    dev = _device()
    model.to(dev).eval()
    xb = torch.from_numpy(X).to(dev)
    yb = torch.from_numpy(y).to(dev)
    out = model(xb)
    return nn.functional.cross_entropy(out, yb, reduction="none").cpu().numpy()


def balanced_class_weights(y: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Inverse-frequency weights so a hospital that sees 6 Chikungunya cases
    does not simply learn to never predict Chikungunya."""
    num_classes = num_classes if num_classes is not None else cfg.NUM_CLASSES
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return (w / w.mean()).astype(np.float32)
