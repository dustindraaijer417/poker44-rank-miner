"""v14 extended feature extraction.

Builds on v1_features (83 dims) + OLD features (239 dims) + adds new:
- Action n-grams (bigrams/trigrams of action types per hand)
- Bet-sizing histograms per chunk (in BB units)
- Per-seat behavior consistency
- Pot-ratio entropy (bots use uniform sizing)
- Streak detection across hands

Total feature dim: ~370.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from neurons.v1_features import CHUNK_FEATURE_NAMES as V1_NAMES, extract_chunk_features as v1_extract
from neurons.feature_extraction import CHUNK_FEATURE_NAMES as OLD_NAMES, extract_chunk_features as old_extract

_BLIND = ("small_blind", "big_blind", "other")
_ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in")


def _safe_entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs) if probs else 0.0


def _action_bigrams(hand):
    actions = [a for a in (hand.get("actions") or [])
               if (a.get("action_type") or "").lower() not in _BLIND]
    counts = Counter()
    for i in range(len(actions) - 1):
        a, b = (actions[i].get("action_type") or "").lower(), (actions[i+1].get("action_type") or "").lower()
        counts[f"{a}_{b}"] += 1
    return counts


_BIGRAM_FEATURES = [
    "fold_fold", "call_call", "check_check", "bet_call", "call_fold",
    "raise_call", "raise_fold", "check_call", "check_fold", "bet_fold",
    "raise_raise", "bet_raise",
]


def _empty_extended_features() -> Dict[str, float]:
    """Returns all extended feature names with 0.0 values."""
    feats = {}
    for bg in _BIGRAM_FEATURES:
        feats[f"bg_{bg}"] = 0.0
    feats["bigram_entropy"] = 0.0
    feats["bigram_unique_ratio"] = 0.0
    for k in ("bet_pct_lt_2bb", "bet_pct_2to5bb", "bet_pct_5to10bb",
              "bet_pct_10to25bb", "bet_pct_gte_25bb", "bet_round_ratio",
              "bet_size_entropy", "pot_rel_bet_entropy", "pot_rel_bet_uniform_score",
              "seat_fold_cv", "seat_bet_cv", "seat_raise_cv",
              "seat_pairwise_dist_mean", "raise_cv", "raise_mode_dominance",
              "fold_per_hand_std", "aggr_per_hand_std", "showdown_rate",
              "max_action_streak"):
        feats[k] = 0.0
    return feats


def _extended_chunk_features(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute extra features on top of v1 + OLD."""
    # Empty chunk: return zeros for all features (computed below).
    if not chunk:
        return _empty_extended_features()

    # Aggregate bigrams across all hands
    all_bigrams = Counter()
    bet_amounts_chunk = []
    seat_actions = {}     # seat -> Counter of action types
    seat_n_actions = {}   # seat -> total
    pot_relative_bets = []
    bet_sizes_in_bb = []
    fold_per_hand = []
    aggression_per_hand = []
    raise_amts = []
    showdown_count = 0

    for hand in chunk:
        actions = [a for a in (hand.get("actions") or [])
                   if (a.get("action_type") or "").lower() not in _BLIND]
        all_bigrams.update(_action_bigrams(hand))

        agg = passive = fold_n = total_n = 0
        for a in actions:
            atype = (a.get("action_type") or "").lower()
            seat = a.get("actor_seat", 0) or 0
            if seat > 0:
                seat_actions.setdefault(seat, Counter())[atype] += 1
                seat_n_actions[seat] = seat_n_actions.get(seat, 0) + 1
            total_n += 1
            if atype in ("bet", "raise", "all_in"):
                agg += 1
                try:
                    amt = float(a.get("normalized_amount_bb") or 0)
                    pb = float(a.get("pot_before") or 0)
                except (TypeError, ValueError):
                    amt = pb = 0
                if amt > 0:
                    bet_amounts_chunk.append(amt)
                    bet_sizes_in_bb.append(amt)
                    if atype == "raise":
                        raise_amts.append(amt)
                    if pb > 0:
                        pot_relative_bets.append(amt / pb)
            elif atype in ("call", "check"):
                passive += 1
            elif atype == "fold":
                fold_n += 1

        if hand.get("outcome", {}).get("showdown"):
            showdown_count += 1
        if total_n > 0:
            fold_per_hand.append(fold_n / total_n)
            aggression_per_hand.append(agg / max(agg + passive, 1))

    n_chunks = max(1, len(chunk))

    feats = {}

    # Bigram frequencies (normalized by total bigrams)
    total_bigrams = max(1, sum(all_bigrams.values()))
    for bg in _BIGRAM_FEATURES:
        feats[f"bg_{bg}"] = all_bigrams.get(bg, 0) / total_bigrams

    # Bigram entropy (more entropy = more variety = human-like)
    feats["bigram_entropy"] = _safe_entropy(list(all_bigrams.values()))
    feats["bigram_unique_ratio"] = len(all_bigrams) / total_bigrams if total_bigrams else 0.0

    # Bet-sizing histograms (% of bets in each bucket)
    if bet_sizes_in_bb:
        arr = np.array(bet_sizes_in_bb)
        n = len(arr)
        feats["bet_pct_lt_2bb"] = float((arr < 2).mean())
        feats["bet_pct_2to5bb"] = float(((arr >= 2) & (arr < 5)).mean())
        feats["bet_pct_5to10bb"] = float(((arr >= 5) & (arr < 10)).mean())
        feats["bet_pct_10to25bb"] = float(((arr >= 10) & (arr < 25)).mean())
        feats["bet_pct_gte_25bb"] = float((arr >= 25).mean())
        # Round-number bias (bots use round bb sizes)
        rounded = np.array([round(x) for x in arr])
        feats["bet_round_ratio"] = float((np.abs(arr - rounded) < 0.5).mean())
        # Bet size entropy
        bins, _ = np.histogram(arr, bins=20)
        feats["bet_size_entropy"] = _safe_entropy(list(bins))
    else:
        for k in ("bet_pct_lt_2bb", "bet_pct_2to5bb", "bet_pct_5to10bb",
                  "bet_pct_10to25bb", "bet_pct_gte_25bb", "bet_round_ratio",
                  "bet_size_entropy"):
            feats[k] = 0.0

    # Pot-relative bet entropy
    if len(pot_relative_bets) >= 3:
        arr = np.array(pot_relative_bets)
        bins, _ = np.histogram(arr, bins=10)
        feats["pot_rel_bet_entropy"] = _safe_entropy(list(bins))
        feats["pot_rel_bet_uniform_score"] = float(np.std(arr) / max(np.mean(arr), 1e-9))
    else:
        feats["pot_rel_bet_entropy"] = 0.0
        feats["pot_rel_bet_uniform_score"] = 0.0

    # Per-seat consistency: do all seats play similarly?
    if seat_actions and len(seat_actions) >= 2:
        seat_action_dists = []
        for s, counts in seat_actions.items():
            total = max(1, sum(counts.values()))
            dist = [counts.get(a, 0) / total for a in _ACTION_TYPES]
            seat_action_dists.append(dist)
        seat_action_dists = np.array(seat_action_dists)
        # Coefficient of variation across seats per action
        feats["seat_fold_cv"] = float(np.std(seat_action_dists[:, 0]) / max(np.mean(seat_action_dists[:, 0]), 1e-9))
        feats["seat_bet_cv"] = float(np.std(seat_action_dists[:, 3]) / max(np.mean(seat_action_dists[:, 3]), 1e-9))
        feats["seat_raise_cv"] = float(np.std(seat_action_dists[:, 4]) / max(np.mean(seat_action_dists[:, 4]), 1e-9))
        # Avg pairwise L2 distance between seat distributions (high = humans, low = bots)
        from itertools import combinations
        dists = [float(np.linalg.norm(np.array(a) - np.array(b))) for a, b in combinations(seat_action_dists, 2)]
        feats["seat_pairwise_dist_mean"] = float(np.mean(dists)) if dists else 0.0
    else:
        for k in ("seat_fold_cv", "seat_bet_cv", "seat_raise_cv", "seat_pairwise_dist_mean"):
            feats[k] = 0.0

    # Raise size consistency
    if len(raise_amts) >= 3:
        arr = np.array(raise_amts)
        feats["raise_cv"] = float(np.std(arr) / max(np.mean(arr), 1e-9))
        # Mode dominance: do bots concentrate on a few sizes?
        rounded = np.round(arr)
        most_common_count = Counter(rounded.tolist()).most_common(1)[0][1] if len(rounded) else 0
        feats["raise_mode_dominance"] = most_common_count / len(arr)
    else:
        feats["raise_cv"] = 0.0
        feats["raise_mode_dominance"] = 0.0

    # Cross-hand consistency
    if len(fold_per_hand) >= 2:
        feats["fold_per_hand_std"] = float(np.std(fold_per_hand))
        feats["aggr_per_hand_std"] = float(np.std(aggression_per_hand))
    else:
        feats["fold_per_hand_std"] = 0.0
        feats["aggr_per_hand_std"] = 0.0

    # Showdown frequency
    feats["showdown_rate"] = showdown_count / n_chunks

    # Action streaks: longest run of same action across hands
    feats["max_action_streak"] = 0.0  # placeholder, expensive to compute
    return feats


EXTENDED_NAMES = list(_extended_chunk_features([]).keys())


# Combine v1 + OLD + extended into final v14 feature vector
V14_NAMES = ([f"v1__{n}" for n in V1_NAMES]
             + [f"old__{n}" for n in OLD_NAMES]
             + [f"ext__{n}" for n in EXTENDED_NAMES])


def extract_v14_features(chunk: List[Dict[str, Any]]) -> np.ndarray:
    """v14 = v1 (83) + OLD (239) + extended (~30) = ~352 features."""
    v1f = v1_extract(chunk)
    oldf = old_extract(chunk)
    ext = _extended_chunk_features(chunk)
    extf = np.array([ext[k] for k in EXTENDED_NAMES], dtype=np.float32)
    return np.concatenate([v1f, oldf, extf]).astype(np.float32)
