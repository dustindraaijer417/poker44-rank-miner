"""V1-aware feature extraction for Poker44 bot detection.

Schema-agnostic features that work on both OLD and V1 chunks:
- All features are RATE-based (no absolute monetary amounts)
- Filters 'other'/blind actions consistently
- Cross-hand consistency signals capture bot regularity
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

import numpy as np

_BLIND_ACTIONS = ("small_blind", "big_blind", "other")
_VOLUNTARY = ("fold", "check", "call", "bet", "raise", "all_in")


def _safe_entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs) if probs else 0.0


def _hand_features(hand: Dict[str, Any]) -> Dict[str, float]:
    """Extract per-hand rate-based features."""
    actions = [a for a in (hand.get("actions") or [])
               if (a.get("action_type") or "").lower() not in _BLIND_ACTIONS]

    counts = Counter()
    streets = Counter()
    bet_amts: List[float] = []
    pot_rel_bets: List[float] = []
    seat_actions: Dict[int, list] = {}
    bigrams: Counter = Counter()
    prev_type = None

    for a in actions:
        atype = (a.get("action_type") or "").lower()
        counts[atype] += 1
        streets[(a.get("street") or "")] += 1
        seat = a.get("actor_seat", 0) or 0
        if seat > 0:
            seat_actions.setdefault(seat, []).append(atype)
        if prev_type:
            bigrams[(prev_type, atype)] += 1
        prev_type = atype
        if atype in ("bet", "raise", "all_in"):
            try:
                amt = float(a.get("normalized_amount_bb") or 0.0)
            except (TypeError, ValueError):
                amt = 0.0
            try:
                pb = float(a.get("pot_before") or 0.0)
            except (TypeError, ValueError):
                pb = 0.0
            if amt > 0:
                bet_amts.append(amt)
                if pb > 0:
                    pot_rel_bets.append(amt / pb)

    n_vol = max(1, sum(counts.get(k, 0) for k in _VOLUNTARY))
    agg_count = counts["bet"] + counts["raise"] + counts["all_in"]
    passive_count = counts["call"] + counts["check"]

    n_streets = sum(1 for s in ("preflop", "flop", "turn", "river") if streets.get(s, 0) > 0)

    if bet_amts:
        bet_arr = np.array(bet_amts)
        bet_mean = float(bet_arr.mean())
        bet_std = float(bet_arr.std())
        bet_cv = bet_std / bet_mean if bet_mean > 1e-9 else 0.0
        # Unique bet sizes — bots use fixed sizing
        rounded = [round(b, 0) for b in bet_amts]
        bet_unique_ratio = len(set(rounded)) / len(bet_amts)
    else:
        bet_cv = 0.0
        bet_unique_ratio = 0.0

    if pot_rel_bets:
        prb = np.array(pot_rel_bets)
        prb_mean = float(prb.mean())
        prb_cv = float(prb.std() / prb_mean) if prb_mean > 1e-9 else 0.0
    else:
        prb_mean = 0.0
        prb_cv = 0.0

    # Action diversity per seat
    if seat_actions:
        seat_diversity = float(np.mean([len(set(v)) for v in seat_actions.values()]))
    else:
        seat_diversity = 0.0

    # Action type entropy
    type_entropy = _safe_entropy([counts.get(k, 0) for k in _VOLUNTARY])
    # Bigram entropy (sequence predictability)
    bigram_entropy = _safe_entropy(list(bigrams.values()))

    return {
        "frac_fold": counts["fold"] / n_vol,
        "frac_check": counts["check"] / n_vol,
        "frac_call": counts["call"] / n_vol,
        "frac_bet": counts["bet"] / n_vol,
        "frac_raise": counts["raise"] / n_vol,
        "frac_allin": counts["all_in"] / n_vol,
        "aggression_ratio": agg_count / max(agg_count + passive_count, 1),
        "n_voluntary": float(n_vol),
        "n_streets": float(n_streets),
        "frac_preflop": streets.get("preflop", 0) / max(sum(streets.values()), 1),
        "frac_postflop": (streets.get("flop", 0) + streets.get("turn", 0) + streets.get("river", 0))
                         / max(sum(streets.values()), 1),
        "bet_cv": bet_cv,
        "bet_unique_ratio": bet_unique_ratio,
        "pot_rel_bet_mean": min(prb_mean, 5.0),  # clip to avoid extremes
        "pot_rel_bet_cv": prb_cv,
        "seat_diversity": seat_diversity,
        "type_entropy": type_entropy,
        "bigram_entropy": bigram_entropy,
    }


HAND_FEATURE_NAMES = list(_hand_features({}).keys())


def extract_chunk_features(chunk: List[Dict[str, Any]]) -> np.ndarray:
    """Aggregate hand features across the chunk + cross-hand consistency."""
    if not chunk:
        return np.zeros(len(CHUNK_FEATURE_NAMES), dtype=np.float32)

    per_hand = [_hand_features(h) for h in chunk]

    # Aggregate each hand feature: mean, std, p25, p75
    arrs = {k: np.array([h[k] for h in per_hand]) for k in HAND_FEATURE_NAMES}

    feats: List[float] = []
    for k in HAND_FEATURE_NAMES:
        a = arrs[k]
        feats.append(float(a.mean()))
        feats.append(float(a.std()))
        feats.append(float(np.percentile(a, 25)))
        feats.append(float(np.percentile(a, 75)))

    # Chunk-level signals
    feats.append(float(len(chunk)))
    # Cross-hand consistency on key features
    for k in ("aggression_ratio", "frac_fold", "frac_bet", "pot_rel_bet_mean", "bet_cv"):
        a = arrs[k]
        feats.append(float(a.std()))
        # Autocorrelation (low for humans, high for bots)
        if len(a) > 1 and a.std() > 1e-9:
            ac = float(np.corrcoef(a[:-1], a[1:])[0, 1])
            if not np.isfinite(ac):
                ac = 0.0
        else:
            ac = 0.0
        feats.append(ac)

    return np.array(feats, dtype=np.float32)


def _build_chunk_feature_names() -> List[str]:
    names: List[str] = []
    for k in HAND_FEATURE_NAMES:
        for stat in ("mean", "std", "p25", "p75"):
            names.append(f"{k}_{stat}")
    names.append("n_hands")
    for k in ("aggression_ratio", "frac_fold", "frac_bet", "pot_rel_bet_mean", "bet_cv"):
        names.append(f"{k}_chunk_std")
        names.append(f"{k}_autocorr")
    return names


CHUNK_FEATURE_NAMES = _build_chunk_feature_names()
