"""Adaptive bot-detection calibration adapted from AceGuard Engine.

Based on Krzysiek99999/aceguard-engine (MIT, 2026 Krzysiek99999), specifically:
- Otsu's method per-batch threshold
- Semantic banding (HUMAN_LOW=0.05, BOT_HI_OUT=0.95, etc.)
- Hard safety cap (max_bot_fraction)
- Anti-FP demotion of weakest bot predictions

This is the calibration approach used by the current top-1 daily winner
(UID 38/109 alternating). They achieve +5.6pp over baseline using this.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Output bands (validator rounds at 0.5: anything ≤ 0.5 = label 0)
HUMAN_LOW = 0.05
HUMAN_HI_OUT = 0.15
LEAN_HUMAN_LOW = 0.30
LEAN_HUMAN_HI = 0.48
LEAN_BOT_LOW = 0.55
LEAN_BOT_HI = 0.72
BOT_LOW = 0.85
BOT_HI_OUT = 0.95


def _otsu_threshold(arr: np.ndarray) -> float:
    """Find Otsu threshold (minimizes intra-class variance)."""
    sorted_vals = np.sort(arr)
    n = len(arr)
    best_t = float(np.median(arr))
    best_var = float("inf")
    for i in range(1, n):
        if sorted_vals[i] == sorted_vals[i - 1]:
            continue
        t = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0
        c0 = arr[arr <= t]
        c1 = arr[arr > t]
        if len(c0) == 0 or len(c1) == 0:
            continue
        w0 = len(c0) / n
        w1 = len(c1) / n
        var_within = w0 * np.var(c0) + w1 * np.var(c1)
        if var_within < best_var:
            best_var = var_within
            best_t = t
    return float(best_t)


def adaptive_safe_calibrate(
    raw_scores: Sequence[float],
    *,
    max_bot_fraction: float = 0.10,
    min_bot_fraction: float = 0.0,
) -> np.ndarray:
    """Adaptive Otsu split + safety cap + (optional) recall floor.

    Reward formula context: `(0.65 * AP + 0.35 * recall) * (1 - FPR)^2`, with
    FPR >= 0.10 hard-zeroing the score. AP uses raw ranking, recall uses
    predictions > 0.5. Otsu alone finds the natural bimodal split — fine for
    AP but tends to under-predict bots when model output is heavily peaked
    on humans, leaving recall near zero.

    `min_bot_fraction` forces a minimum number of top-ranked chunks above the
    bot band so recall doesn't collapse. The cost is FPR risk if the top
    forced picks are actually humans; calling miners must keep this below
    the 10% cliff (so min_bot_fraction * batch_size <= expected_humans * 0.10).

    Args:
        raw_scores: Raw model probabilities for the batch.
        max_bot_fraction: Maximum fraction of batch that can be bot.
        min_bot_fraction: Minimum fraction of batch that must be bot —
            forces top-N picks above 0.5 even if Otsu finds fewer.

    Returns:
        np.array of calibrated scores in [0.05, 0.95], rank-preserving.
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Step 1: Otsu split finds natural threshold
    threshold = _otsu_threshold(arr)
    is_bot = arr > threshold
    n_bot = int(is_bot.sum())

    # Step 2: Recall floor — promote top-ranked chunks if Otsu under-predicts
    min_bots = int(n * min_bot_fraction)
    if n_bot < min_bots:
        sorted_idx = np.argsort(-arr)  # descending
        is_bot = np.zeros(n, dtype=bool)
        is_bot[sorted_idx[:min_bots]] = True
        n_bot = min_bots

    # Step 3: Hard safety cap
    max_bots = int(n * max_bot_fraction)
    if n_bot > max_bots:
        bot_idx = np.where(is_bot)[0]
        bot_scores = arr[bot_idx]
        sorted_by_score = bot_idx[np.argsort(bot_scores)]
        to_demote = set(sorted_by_score[: n_bot - max_bots].tolist())
        is_bot = np.array([(i in set(np.where(is_bot)[0]) and i not in to_demote) for i in range(n)])
        n_bot = int(is_bot.sum())

    n_human = n - n_bot

    # Step 3: Output rank-preserving scores in semantic bands
    out = np.empty(n, dtype=np.float64)
    bot_indices = np.where(is_bot)[0]
    human_indices = np.where(~is_bot)[0]

    # Bots: top of ranking gets BOT_HI_OUT, bottom gets BOT_LOW
    if n_bot > 0:
        bot_order = bot_indices[np.argsort(-arr[bot_indices])]
        for rank, idx in enumerate(bot_order):
            t_frac = rank / max(n_bot - 1, 1)
            out[idx] = BOT_HI_OUT - t_frac * (BOT_HI_OUT - BOT_LOW)

    # Humans: lowest raw scores get HUMAN_LOW, highest get HUMAN_HI_OUT
    if n_human > 0:
        human_order = human_indices[np.argsort(arr[human_indices])]
        for rank, idx in enumerate(human_order):
            t_frac = rank / max(n_human - 1, 1)
            out[idx] = HUMAN_LOW + t_frac * (HUMAN_HI_OUT - HUMAN_LOW)

    return out
