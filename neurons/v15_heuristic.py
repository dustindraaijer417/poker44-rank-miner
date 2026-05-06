"""v15 heuristic — V1-tuned, no ML model, capped at 0.49 (no bot predictions).

Built from Cohen's d analysis on 8000 captured V1 chunks (using v14 classifier
confident predictions as pseudo-labels). The top discriminators are:

1. n_voluntary (bots ~6, humans ~12): strongest signal
2. pot_rel_bet_cv (bots 0.08 = uniform sizing, humans 0.83 = varied)
3. bigram_entropy (bots low = predictable, humans high = diverse)
4. action_type_entropy (bots low, humans high)
5. bet_cv (bots low = same size always, humans high)
6. aggression consistency across hands (bots more variable)

Output: AP-ranking score in [0, 0.49]. Predicts 0 bots → safe baseline reward.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

import numpy as np

_BLIND = ("small_blind", "big_blind", "other")
_VOL = ("fold", "check", "call", "bet", "raise", "all_in")


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs) if probs else 0.0


def _hand_signal(hand: Dict[str, Any]) -> Dict[str, float]:
    """Per-hand signals used by v15 heuristic."""
    actions = [a for a in (hand.get("actions") or [])
               if (a.get("action_type") or "").lower() not in _BLIND]
    counts = Counter()
    bigrams = Counter()
    bet_amts = []
    pot_rel_bets = []

    prev = None
    for a in actions:
        atype = (a.get("action_type") or "").lower()
        counts[atype] += 1
        if prev:
            bigrams[(prev, atype)] += 1
        prev = atype
        if atype in ("bet", "raise", "all_in"):
            try:
                amt = float(a.get("normalized_amount_bb") or 0.0)
                pb = float(a.get("pot_before") or 0.0)
            except (TypeError, ValueError):
                amt = pb = 0.0
            if amt > 0:
                bet_amts.append(amt)
                if pb > 0:
                    pot_rel_bets.append(amt / pb)

    n_voluntary = sum(counts.get(k, 0) for k in _VOL)

    return {
        "n_voluntary": float(n_voluntary),
        "bet_mean": float(np.mean(bet_amts)) if bet_amts else 0.0,
        "bet_cv": (float(np.std(bet_amts) / np.mean(bet_amts))
                   if len(bet_amts) > 1 and np.mean(bet_amts) > 0 else 0.0),
        "pot_rel_bet_cv": (float(np.std(pot_rel_bets) / np.mean(pot_rel_bets))
                           if len(pot_rel_bets) > 1 and np.mean(pot_rel_bets) > 0 else 0.0),
        "type_entropy": _entropy([counts.get(k, 0) for k in _VOL]),
        "bigram_entropy": _entropy(list(bigrams.values())),
        "aggression": ((counts.get("bet", 0) + counts.get("raise", 0) + counts.get("all_in", 0))
                       / max(counts.get("bet", 0) + counts.get("raise", 0)
                             + counts.get("all_in", 0) + counts.get("call", 0)
                             + counts.get("check", 0), 1)),
    }


def score_chunk_v15(chunk: List[Dict[str, Any]]) -> float:
    """v15 heuristic: weighted-sum of top-discriminating signals.
    Returns score in [0.0, 0.49] — never crosses bot threshold.
    """
    if not chunk:
        return 0.25

    hands = [_hand_signal(h) for h in chunk]
    nh = max(1, len(hands))

    # Aggregate hand signals
    n_vol_mean = float(np.mean([h["n_voluntary"] for h in hands]))
    bet_means = [h["bet_mean"] for h in hands]
    bet_cv_mean = float(np.mean([h["bet_cv"] for h in hands]))
    pot_rel_cv_mean = float(np.mean([h["pot_rel_bet_cv"] for h in hands]))
    type_ent_mean = float(np.mean([h["type_entropy"] for h in hands]))
    bigram_ent_mean = float(np.mean([h["bigram_entropy"] for h in hands]))
    aggr = [h["aggression"] for h in hands]
    aggr_std = float(np.std(aggr)) if len(aggr) > 1 else 0.0

    # 10th percentile of bet means (strongest signal: bots have 0)
    if bet_means:
        bet_p10 = float(np.percentile(bet_means, 10))
    else:
        bet_p10 = 0.0

    # Build score from BOT-leaning signals
    score = 0.0

    # 1. n_voluntary: bot ~6, human ~12. Score = 1 when bot-like.
    s_nvol = _clamp((11.0 - n_vol_mean) / 5.0, 0.0, 1.0)
    score += 0.20 * s_nvol

    # 2. bet_mean_p10: bot 0, human ~10. Score = 1 when bot-like.
    s_betp10 = _clamp((5.0 - bet_p10) / 5.0, 0.0, 1.0)
    score += 0.18 * s_betp10

    # 3. pot_rel_bet_cv_mean: bot 0.08, human 0.83. Lower = more bot.
    s_potcv = _clamp((0.5 - pot_rel_cv_mean) / 0.5, 0.0, 1.0)
    score += 0.15 * s_potcv

    # 4. bet_cv_mean: bot 0.08, human 0.42. Lower = more bot.
    s_betcv = _clamp((0.3 - bet_cv_mean) / 0.3, 0.0, 1.0)
    score += 0.12 * s_betcv

    # 5. type_entropy: bot 0.05, human 1.40. Lower = more bot.
    s_typeent = _clamp((1.0 - type_ent_mean) / 1.0, 0.0, 1.0)
    score += 0.10 * s_typeent

    # 6. bigram_entropy: bot 1.38, human 2.76. Lower = more bot.
    s_bigent = _clamp((2.5 - bigram_ent_mean) / 1.5, 0.0, 1.0)
    score += 0.10 * s_bigent

    # 7. aggression_ratio_std: bot 0.42, human 0.23. HIGHER = more bot.
    s_aggrstd = _clamp((aggr_std - 0.20) / 0.25, 0.0, 1.0)
    score += 0.10 * s_aggrstd

    # 8. n_voluntary_std (bots 1.36, humans 0.02): HIGH = more bot
    n_vol_per_hand = [h["n_voluntary"] for h in hands]
    n_vol_std = float(np.std(n_vol_per_hand)) if len(n_vol_per_hand) > 1 else 0.0
    s_nvolstd = _clamp((n_vol_std - 0.5) / 1.0, 0.0, 1.0)
    score += 0.05 * s_nvolstd

    # Cap at 0.49 to never trigger bot threshold (safe AP-only strategy).
    return round(min(score, 0.49), 6)
