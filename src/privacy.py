"""
src/privacy.py
==============
Differential Privacy for FedMed-FL, in two flavours, plus a real RDP
accountant so the manuscript can report an honest (epsilon, delta) pair
instead of hand-waving "we added noise".

    MODE A — RECORD-LEVEL DP  (default, recommended)
    ------------------------------------------------
    DP-SGD inside each hospital's local training loop (Abadi et al., 2016):
        1. compute PER-EXAMPLE gradients
        2. clip each one to L2 norm C          -> sensitivity of one patient = C
        3. sum, add N(0, sigma^2 C^2 I), divide by batch size
    Guarantee: "no single PATIENT RECORD measurably changes what is released."
    That is exactly the guarantee HIPAA / DISHA care about, and privacy
    amplification by subsampling keeps epsilon small enough to publish.

    MODE B — CLIENT-LEVEL DP  (provided for the ablation)
    ----------------------------------------------------
    Clip + noise the whole model update before transmitting it.
    Guarantee: "no single HOSPITAL measurably changes the global model."
    IMPORTANT FINDING FOR YOUR PAPER: with only K = 3 silos this destroys
    utility. Averaged noise per coordinate is sigma*C/K while signal per
    coordinate is only about C/sqrt(d); with d ~ 14,000 and K = 3 the noise
    dominates by orders of magnitude. Run `--dp-mode client` and show the
    collapse — a defensible negative result and a genuine contribution to the
    cross-silo FL literature.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

import config as cfg


# ===========================================================================
# MODE A : RECORD-LEVEL DP-SGD
# ===========================================================================
def make_per_sample_grad_fn(model: nn.Module, class_weights: torch.Tensor | None = None):
    """Returns f(params, buffers, X, y) -> dict of per-sample gradients.

    Uses `torch.func.vmap(grad(...))`, which vectorises the per-example
    backward pass — the same technique Opacus uses, roughly 100x faster than
    a naive microbatch loop.

    `class_weights` re-weights rare diagnoses. It does NOT weaken the privacy
    guarantee: every per-example gradient is clipped to the same norm C
    afterwards, so one patient's sensitivity is C regardless of their class.
    """
    def _loss(params, buffers, x, y):
        out = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return nn.functional.cross_entropy(out, y.unsqueeze(0), weight=class_weights)

    return vmap(grad(_loss), in_dims=(None, None, 0, 0), randomness="different")


def clip_and_noise_per_sample(per_sample_grads: dict, clip_norm: float,
                              noise_multiplier: float, batch_size: int,
                              device) -> dict:
    """Core DP-SGD step: per-example clip -> sum -> Gaussian noise -> average."""
    # 1. per-example L2 norm across ALL parameter tensors
    sq = None
    for g in per_sample_grads.values():
        s = g.pow(2).flatten(1).sum(1)
        sq = s if sq is None else sq + s
    norms = sq.sqrt()

    # 2. one clip factor per example (never scales up, only down)
    factor = (clip_norm / (norms + 1e-6)).clamp(max=1.0)

    # 3. clipped sum + calibrated noise, then average
    out = {}
    for name, g in per_sample_grads.items():
        shape = [-1] + [1] * (g.dim() - 1)
        summed = (g * factor.view(shape)).sum(0)
        if noise_multiplier > 0:
            summed = summed + torch.normal(
                0.0, noise_multiplier * clip_norm, size=summed.shape, device=device)
        out[name] = summed / batch_size
    return out


# ===========================================================================
# MODE B : CLIENT-LEVEL DP ON THE TRANSMITTED UPDATE
# ===========================================================================
def flat_l2_norm(update: list[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.square(u))) for u in update)))


def clip_update(update: list[np.ndarray], clip_norm: float
                ) -> tuple[list[np.ndarray], float]:
    norm = flat_l2_norm(update)
    scale = min(1.0, clip_norm / (norm + 1e-12))
    return [u * scale for u in update], norm


def privatise(global_w: list[np.ndarray], local_w: list[np.ndarray],
              clip_norm: float = cfg.DP_CLIP_NORM,
              noise_multiplier: float = cfg.DP_NOISE_MULTIPLIER,
              num_clients: int = cfg.NUM_CLIENTS,
              rng: np.random.Generator | None = None
              ) -> tuple[list[np.ndarray], float]:
    """Clip the update to C, add N(0,(sigma*C/K)^2), return weights to send.

    Dividing by K mirrors standard DP-FedAvg, where the server adds
    N(0, sigma^2 C^2) once to the SUM and then divides by K.
    """
    rng = rng or np.random.default_rng()
    update = [l - g for l, g in zip(local_w, global_w)]
    update, raw_norm = clip_update(update, clip_norm)
    std = noise_multiplier * clip_norm / max(num_clients, 1)
    if std > 0:
        update = [u + rng.normal(0.0, std, size=u.shape).astype(u.dtype) for u in update]
    return [g + u for g, u in zip(global_w, update)], raw_norm


# ===========================================================================
# RDP ACCOUNTANT
# ===========================================================================
_INT_ALPHAS = list(range(2, 129))
_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + _INT_ALPHAS


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _rdp_subsampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of the Poisson-subsampled Gaussian mechanism at INTEGER order alpha
    (Mironov, Talwar & Zhang, 2019), computed in log-space for stability:

        eps(a) = 1/(a-1) * log( SUM_k C(a,k) (1-q)^(a-k) q^k exp(k(k-1)/(2 s^2)) )
    """
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2.0 * sigma ** 2)

    log_terms = []
    for k in range(alpha + 1):
        t = (_log_comb(alpha, k)
             + (alpha - k) * math.log1p(-q)
             + k * math.log(q)
             + (k * (k - 1)) / (2.0 * sigma ** 2))
        log_terms.append(t)
    m = max(log_terms)
    total = m + math.log(sum(math.exp(t - m) for t in log_terms))
    return total / (alpha - 1)


def _rdp_to_dp(rdp: float, alpha: float, delta: float) -> float:
    """Tight RDP -> (eps, delta)-DP conversion (Canonne, Kamath & Steinke, 2020)."""
    if alpha <= 1:
        return float("inf")
    return rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)


def epsilon_record_level(steps: int, sample_rate: float, noise_multiplier: float,
                         delta: float = cfg.DP_DELTA) -> float:
    """(eps, delta) for `steps` DP-SGD iterations at sampling rate q.

    `steps` must count EVERY local SGD step a patient's record could have been
    sampled into, across ALL federated rounds.
    """
    if noise_multiplier <= 0 or sample_rate <= 0 or steps <= 0:
        return float("inf")
    best = float("inf")
    for a in _INT_ALPHAS:
        rdp = steps * _rdp_subsampled_gaussian(sample_rate, noise_multiplier, a)
        best = min(best, _rdp_to_dp(rdp, a, delta))
    return float(max(best, 0.0))


def epsilon_client_level(num_rounds: int, noise_multiplier: float,
                         delta: float = cfg.DP_DELTA) -> float:
    """(eps, delta) for `num_rounds` releases of the UN-subsampled Gaussian
    mechanism. All 3 silos participate every round, so there is no privacy
    amplification and epsilon grows quickly — that is the point of the ablation."""
    if noise_multiplier <= 0:
        return float("inf")
    best = float("inf")
    for a in _ALPHAS:
        if a <= 1:
            continue
        best = min(best, _rdp_to_dp(num_rounds * a / (2.0 * noise_multiplier ** 2), a, delta))
    return float(max(best, 0.0))


def epsilon_worst_client(per_client: dict, noise_multiplier: float,
                         delta: float = cfg.DP_DELTA) -> tuple[float, str]:
    """Epsilon must be computed PER CLIENT and the WORST one reported.

    Each hospital has its own dataset size, hence its own sampling rate q and
    its own step count. Mixing one hospital's q with another's step count
    (e.g. `max(q)` with `max(steps)`) produces a number that describes no
    actual participant. Returns (worst_epsilon, worst_client_id).
    """
    worst, who = 0.0, ""
    for cid, (steps, q) in per_client.items():
        e = epsilon_record_level(steps, q, noise_multiplier, delta)
        if e > worst:
            worst, who = e, cid
    return worst, who


def privacy_report(mode: str, *, steps: int = 0, sample_rate: float = 0.0,
                   per_client: dict | None = None,
                   num_rounds: int = cfg.NUM_ROUNDS,
                   noise_multiplier: float = cfg.DP_NOISE_MULTIPLIER,
                   clip_norm: float = cfg.DP_CLIP_NORM,
                   delta: float = cfg.DP_DELTA) -> dict:
    if mode == "none" or noise_multiplier <= 0:
        return {"mode": "none", "epsilon": None,
                "note": "No formal privacy guarantee — this is the utility-ceiling baseline."}

    worst_client = None
    if mode == "record":
        if per_client:
            eps, worst_client = epsilon_worst_client(per_client, noise_multiplier, delta)
            steps, sample_rate = per_client[worst_client]
        else:
            eps = epsilon_record_level(steps, sample_rate, noise_multiplier, delta)
        unit = "one patient record"
    else:
        eps = epsilon_client_level(num_rounds, noise_multiplier, delta)
        unit = "one hospital's entire participation"

    return {
        "mode": mode,
        "protects": unit,
        "mechanism": ("Per-example clipped gradients + Gaussian noise (DP-SGD)"
                      if mode == "record"
                      else "Clipped model update + Gaussian noise (DP-FedAvg)"),
        "clip_norm_C": clip_norm,
        "noise_multiplier_sigma": noise_multiplier,
        "sample_rate_q": round(sample_rate, 5) if mode == "record" else None,
        "steps": steps if mode == "record" else num_rounds,
        "delta": delta,
        "epsilon": round(eps, 3),
        "worst_case_client": worst_client,
    }


if __name__ == "__main__":
    n, B = 2352, cfg.BATCH_SIZE
    q = B / n
    steps = cfg.NUM_ROUNDS * cfg.LOCAL_EPOCHS * math.ceil(n / B)
    print(f"RECORD-level DP-SGD   n={n}  B={B}  q={q:.4f}  steps={steps}  delta={cfg.DP_DELTA}")
    print(f"{'sigma':>8} {'epsilon':>12}")
    for s in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        print(f"{s:>8.2f} {epsilon_record_level(steps, q, s):>12.2f}")

    print(f"\nCLIENT-level DP-FedAvg   T={cfg.NUM_ROUNDS} rounds (no amplification)")
    print(f"{'sigma':>8} {'epsilon':>12}")
    for s in [0.5, 1.0, 2.0, 3.0, 5.0]:
        print(f"{s:>8.2f} {epsilon_client_level(cfg.NUM_ROUNDS, s):>12.2f}")
