"""Feature extraction from sanitized poker hand chunks for bot detection.

Extracts features that survive the validator's sanitization:
- actions[].street, action_type, normalized_amount_bb, pot_before, pot_after
- players[].starting_stack
- streets[] (board progression)

NOTE: outcome.total_pot and outcome.rake are ZEROED by sanitization,
so we derive pot information from actions[].pot_after instead.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List


def _safe_entropy(counts: List[int]) -> float:
    """Shannon entropy of a discrete distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _extract_hand_features(hand: Dict[str, Any]) -> Dict[str, float]:
    """Extract features from a single sanitized hand."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []

    n_actions = len(actions)

    # --- Street distribution of the 12 sampled actions ---
    street_counts = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
    for action in actions:
        street = action.get("street", "")
        if street in street_counts:
            street_counts[street] += 1

    total_street_actions = sum(street_counts.values()) or 1
    streets_reached = sum(1 for v in street_counts.values() if v > 0)

    frac_preflop = street_counts["preflop"] / total_street_actions
    frac_flop = street_counts["flop"] / total_street_actions
    frac_turn = street_counts["turn"] / total_street_actions
    frac_river = street_counts["river"] / total_street_actions

    reached_flop = 1.0 if street_counts["flop"] > 0 else 0.0
    reached_turn = 1.0 if street_counts["turn"] > 0 else 0.0
    reached_river = 1.0 if street_counts["river"] > 0 else 0.0

    # --- Starting stacks ---
    stacks = [p.get("starting_stack", 0.0) or 0.0 for p in players]
    active_stacks = [s for s in stacks if s > 0]
    n_active_players = len(active_stacks)

    if active_stacks:
        stack_mean = float(np.mean(active_stacks))
        stack_std = float(np.std(active_stacks))
        stack_min = min(active_stacks)
        stack_max = max(active_stacks)
        stack_range = stack_max - stack_min
        stack_cv = stack_std / stack_mean if stack_mean > 0 else 0.0
    else:
        stack_mean = stack_std = stack_min = stack_max = stack_range = stack_cv = 0.0

    # --- Action type counts (key bot signal!) ---
    type_counts = {"fold": 0, "check": 0, "call": 0, "bet": 0, "raise": 0,
                   "all_in": 0, "small_blind": 0, "big_blind": 0}
    for action in actions:
        atype = action.get("action_type", "")
        if atype in type_counts:
            type_counts[atype] += 1

    voluntary_actions = max(n_actions - type_counts["small_blind"] - type_counts["big_blind"], 1)
    frac_fold = type_counts["fold"] / voluntary_actions
    frac_check = type_counts["check"] / voluntary_actions
    frac_call = type_counts["call"] / voluntary_actions
    frac_bet = type_counts["bet"] / voluntary_actions
    frac_raise = type_counts["raise"] / voluntary_actions
    frac_allin = type_counts["all_in"] / voluntary_actions

    # Aggression = (bet + raise + all_in) / (bet + raise + all_in + call + check)
    aggressive = type_counts["bet"] + type_counts["raise"] + type_counts["all_in"]
    passive = type_counts["call"] + type_counts["check"]
    aggression_ratio = aggressive / max(aggressive + passive, 1)

    # --- Action type entropy (bots tend to have less diverse action patterns) ---
    action_type_entropy = _safe_entropy([
        type_counts["fold"], type_counts["check"], type_counts["call"],
        type_counts["bet"], type_counts["raise"], type_counts["all_in"],
    ])

    # --- Per-street aggression breakdown (key discriminator!) ---
    street_aggressive = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
    street_passive = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
    street_bets = {"preflop": [], "flop": [], "turn": [], "river": []}
    for action in actions:
        street = action.get("street", "")
        atype = action.get("action_type", "")
        if street not in street_aggressive:
            continue
        if atype in ("bet", "raise", "all_in"):
            street_aggressive[street] += 1
            amt = float(action.get("normalized_amount_bb", 0.0) or 0.0)
            if amt > 0:
                street_bets[street].append(amt)
        elif atype in ("call", "check"):
            street_passive[street] += 1

    # Per-street aggression ratios
    def _street_aggr(street):
        a = street_aggressive[street]
        p = street_passive[street]
        return a / max(a + p, 1)

    preflop_aggression = _street_aggr("preflop")
    flop_aggression = _street_aggr("flop")
    turn_aggression = _street_aggr("turn")
    river_aggression = _street_aggr("river")

    # Aggression change across streets (bots may be consistent, humans vary)
    aggr_values = [preflop_aggression, flop_aggression, turn_aggression, river_aggression]
    aggr_active = [v for v, sc in zip(aggr_values, street_counts.values()) if sc > 0]
    aggression_street_std = float(np.std(aggr_active)) if len(aggr_active) > 1 else 0.0

    # --- Bet sizing features (from normalized_amount_bb) ---
    bet_amounts = []
    for action in actions:
        atype = action.get("action_type", "")
        if atype in ("bet", "raise", "all_in"):
            amt = float(action.get("normalized_amount_bb", 0.0) or 0.0)
            if amt > 0:
                bet_amounts.append(amt)

    if bet_amounts:
        bet_arr = np.array(bet_amounts)
        bet_mean = float(np.mean(bet_arr))
        bet_std = float(np.std(bet_arr))
        bet_cv = bet_std / bet_mean if bet_mean > 0 else 0.0
        bet_min = float(np.min(bet_arr))
        bet_max = float(np.max(bet_arr))
        bet_median = float(np.median(bet_arr))
        bet_skew = float(((bet_arr - bet_mean) ** 3).mean() / (bet_std ** 3)) if bet_std > 1e-9 else 0.0
        bet_kurtosis = float(((bet_arr - bet_mean) ** 4).mean() / (bet_std ** 4) - 3.0) if bet_std > 1e-9 else 0.0
        # Bet sizing quantization: fraction of bets in common sizing buckets
        n_bets = len(bet_amounts)
        bet_frac_small = sum(1 for b in bet_amounts if b <= 2.0) / n_bets
        bet_frac_medium = sum(1 for b in bet_amounts if 2.0 < b <= 6.0) / n_bets
        bet_frac_large = sum(1 for b in bet_amounts if b > 6.0) / n_bets
        # Unique bet sizes (normalized) — bots often use fixed sizing
        rounded_bets = [round(b, 1) for b in bet_amounts]
        bet_unique_ratio = len(set(rounded_bets)) / n_bets
    else:
        bet_mean = bet_std = bet_cv = bet_min = bet_max = 0.0
        bet_median = bet_skew = bet_kurtosis = 0.0
        bet_frac_small = bet_frac_medium = bet_frac_large = 0.0
        bet_unique_ratio = 0.0

    # --- Pot progression features (from pot_before/pot_after in actions) ---
    pot_values = []
    for action in actions:
        pb = float(action.get("pot_before", 0.0) or 0.0)
        pa = float(action.get("pot_after", 0.0) or 0.0)
        if pa > 0:
            pot_values.append(pa)
        elif pb > 0:
            pot_values.append(pb)

    if pot_values:
        final_pot = pot_values[-1] if pot_values else 0.0
        pot_growth = pot_values[-1] / pot_values[0] if pot_values[0] > 0 else 0.0
        pot_mean = float(np.mean(pot_values))
        pot_std = float(np.std(pot_values))
    else:
        final_pot = pot_growth = pot_mean = pot_std = 0.0

    # Pot-relative bet sizing (bet as fraction of pot)
    pot_relative_bets = []
    for action in actions:
        atype = action.get("action_type", "")
        if atype in ("bet", "raise", "all_in"):
            amt = float(action.get("normalized_amount_bb", 0.0) or 0.0)
            pb = float(action.get("pot_before", 0.0) or 0.0)
            if amt > 0 and pb > 0:
                pot_relative_bets.append(amt / pb)

    if pot_relative_bets:
        prb_arr = np.array(pot_relative_bets)
        pot_rel_bet_mean = float(np.mean(prb_arr))
        pot_rel_bet_std = float(np.std(prb_arr))
        pot_rel_bet_cv = pot_rel_bet_std / pot_rel_bet_mean if pot_rel_bet_mean > 0 else 0.0
        # Pot-relative sizing regularity (bots tend to use fixed pot ratios)
        pot_rel_bet_range = float(np.max(prb_arr) - np.min(prb_arr))
    else:
        pot_rel_bet_mean = pot_rel_bet_std = pot_rel_bet_cv = 0.0
        pot_rel_bet_range = 0.0

    # --- Action sequence pattern ---
    street_transitions = 0
    prev_street = None
    for action in actions:
        s = action.get("street", "")
        if prev_street is not None and s != prev_street:
            street_transitions += 1
        prev_street = s

    # Consecutive same-street actions at the end (padding signal)
    tail_same = 0
    if actions:
        last_street = actions[-1].get("street", "")
        for a in reversed(actions):
            if a.get("street", "") == last_street:
                tail_same += 1
            else:
                break
    tail_same_frac = tail_same / max(n_actions, 1)

    # --- Action sequence entropy (sequence predictability) ---
    # Encode action sequence as type transitions
    action_bigrams: Dict[str, int] = {}
    for i in range(len(actions) - 1):
        t1 = actions[i].get("action_type", "")
        t2 = actions[i + 1].get("action_type", "")
        bigram = f"{t1}_{t2}"
        action_bigrams[bigram] = action_bigrams.get(bigram, 0) + 1
    action_sequence_entropy = _safe_entropy(list(action_bigrams.values()))

    # --- Per-seat action diversity (do different seats behave differently?) ---
    seat_action_types: Dict[int, set] = {}
    seat_action_counts: Dict[int, int] = {}
    for action in actions:
        seat = action.get("actor_seat", 0)
        atype = action.get("action_type", "")
        if seat > 0 and atype:
            seat_action_types.setdefault(seat, set()).add(atype)
            seat_action_counts[seat] = seat_action_counts.get(seat, 0) + 1

    if seat_action_types:
        action_diversity_mean = float(np.mean([len(v) for v in seat_action_types.values()]))
    else:
        action_diversity_mean = 0.0

    # Number of unique actors
    unique_actors = len(seat_action_types)

    # Pot relative to average stack (using derived pot)
    pot_to_stack = final_pot / stack_mean if stack_mean > 0 else 0.0

    # --- Actions per player (concentration) ---
    if seat_action_counts:
        counts_arr = np.array(list(seat_action_counts.values()), dtype=float)
        actions_per_player_std = float(np.std(counts_arr))
        actions_per_player_cv = float(np.std(counts_arr) / np.mean(counts_arr)) if np.mean(counts_arr) > 0 else 0.0
    else:
        actions_per_player_std = actions_per_player_cv = 0.0

    # --- Raise-to-bet ratio (escalation pattern) ---
    n_raises = type_counts["raise"]
    n_bets_only = type_counts["bet"]
    raise_to_bet_ratio = n_raises / max(n_raises + n_bets_only, 1)

    return {
        "n_actions": float(n_actions),
        "frac_preflop": frac_preflop,
        "frac_flop": frac_flop,
        "frac_turn": frac_turn,
        "frac_river": frac_river,
        "streets_reached": float(streets_reached),
        "reached_flop": reached_flop,
        "reached_turn": reached_turn,
        "reached_river": reached_river,
        "n_active_players": float(n_active_players),
        "stack_mean": stack_mean,
        "stack_std": stack_std,
        "stack_min": stack_min,
        "stack_max": stack_max,
        "stack_range": stack_range,
        "stack_cv": stack_cv,
        "street_transitions": float(street_transitions),
        "tail_same_frac": tail_same_frac,
        # Action type features
        "frac_fold": frac_fold,
        "frac_check": frac_check,
        "frac_call": frac_call,
        "frac_bet": frac_bet,
        "frac_raise": frac_raise,
        "frac_allin": frac_allin,
        "aggression_ratio": aggression_ratio,
        "action_type_entropy": action_type_entropy,
        # Per-street aggression
        "preflop_aggression": preflop_aggression,
        "flop_aggression": flop_aggression,
        "turn_aggression": turn_aggression,
        "river_aggression": river_aggression,
        "aggression_street_std": aggression_street_std,
        # Bet sizing features
        "bet_mean": bet_mean,
        "bet_std": bet_std,
        "bet_cv": bet_cv,
        "bet_min": bet_min,
        "bet_max": bet_max,
        "bet_median": bet_median,
        "bet_skew": bet_skew,
        "bet_kurtosis": bet_kurtosis,
        "bet_frac_small": bet_frac_small,
        "bet_frac_medium": bet_frac_medium,
        "bet_frac_large": bet_frac_large,
        "bet_unique_ratio": bet_unique_ratio,
        # Pot progression (derived from actions, NOT zeroed outcome)
        "final_pot": final_pot,
        "pot_growth": pot_growth,
        "pot_mean": pot_mean,
        "pot_std": pot_std,
        # Pot-relative bet sizing
        "pot_rel_bet_mean": pot_rel_bet_mean,
        "pot_rel_bet_std": pot_rel_bet_std,
        "pot_rel_bet_cv": pot_rel_bet_cv,
        "pot_rel_bet_range": pot_rel_bet_range,
        # Action sequence features
        "action_sequence_entropy": action_sequence_entropy,
        # Per-seat behavior
        "action_diversity_mean": action_diversity_mean,
        "unique_actors": float(unique_actors),
        "pot_to_stack": pot_to_stack,
        "actions_per_player_std": actions_per_player_std,
        "actions_per_player_cv": actions_per_player_cv,
        # Escalation
        "raise_to_bet_ratio": raise_to_bet_ratio,
    }


HAND_FEATURE_NAMES = list(_extract_hand_features({}).keys())
N_HAND_FEATURES = len(HAND_FEATURE_NAMES)


def extract_single_hand_features(hand: Dict[str, Any]) -> np.ndarray:
    """Extract features from a single hand as a flat numpy array.

    Used when chunks contain only 1 hand (provider_runtime mode).
    Returns the 37 hand-level features directly without aggregation.
    """
    feats = _extract_hand_features(hand)
    return np.array([feats[k] for k in HAND_FEATURE_NAMES], dtype=np.float32)


SINGLE_HAND_FEATURE_NAMES = list(HAND_FEATURE_NAMES)


def extract_chunk_features(chunk: List[Dict[str, Any]]) -> np.ndarray:
    """Extract aggregated features from a chunk of sanitized hands.

    Returns a 1D numpy array of features suitable for classification.
    """
    if not chunk:
        return np.zeros(len(CHUNK_FEATURE_NAMES), dtype=np.float32)

    hand_features_list = [_extract_hand_features(hand) for hand in chunk]

    # Convert to arrays per feature
    feature_arrays = {}
    for key in HAND_FEATURE_NAMES:
        feature_arrays[key] = np.array(
            [hf[key] for hf in hand_features_list], dtype=np.float64
        )

    # Aggregate: mean, std, and coefficient of variation for each hand-level feature
    chunk_feats = []
    for key in HAND_FEATURE_NAMES:
        arr = feature_arrays[key]
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        cv_val = std_val / mean_val if abs(mean_val) > 1e-9 else 0.0
        chunk_feats.extend([mean_val, std_val, cv_val])

    # Additional chunk-level features
    n_hands = float(len(chunk))
    chunk_feats.append(n_hands)

    # Correlation between features across hands (captures consistency)
    # Bots from same profile should show correlated behavior
    if len(hand_features_list) > 2:
        frac_preflop_arr = feature_arrays["frac_preflop"]
        aggression_arr = feature_arrays["aggression_ratio"]
        bet_mean_arr = feature_arrays["bet_mean"]
        final_pot_arr = feature_arrays["final_pot"]
        frac_fold_arr = feature_arrays["frac_fold"]
        frac_check_arr = feature_arrays["frac_check"]
        pot_rel_bet_arr = feature_arrays["pot_rel_bet_mean"]

        # Autocorrelation of bet sizes (bots may show more regular patterns)
        if len(bet_mean_arr) > 1 and np.std(bet_mean_arr) > 1e-9:
            bet_autocorr = float(np.corrcoef(bet_mean_arr[:-1], bet_mean_arr[1:])[0, 1])
            if np.isnan(bet_autocorr):
                bet_autocorr = 0.0
        else:
            bet_autocorr = 0.0

        # Autocorrelation of final pot (regularity signal)
        if len(final_pot_arr) > 1 and np.std(final_pot_arr) > 1e-9:
            pot_autocorr = float(np.corrcoef(final_pot_arr[:-1], final_pot_arr[1:])[0, 1])
            if np.isnan(pot_autocorr):
                pot_autocorr = 0.0
        else:
            pot_autocorr = 0.0

        # Correlation between preflop fraction and aggression (behavioral coupling)
        if np.std(frac_preflop_arr) > 1e-9 and np.std(aggression_arr) > 1e-9:
            preflop_aggr_corr = float(np.corrcoef(frac_preflop_arr, aggression_arr)[0, 1])
            if np.isnan(preflop_aggr_corr):
                preflop_aggr_corr = 0.0
        else:
            preflop_aggr_corr = 0.0

        # Correlation between fold rate and bet sizing
        if np.std(frac_fold_arr) > 1e-9 and np.std(bet_mean_arr) > 1e-9:
            fold_bet_corr = float(np.corrcoef(frac_fold_arr, bet_mean_arr)[0, 1])
            if np.isnan(fold_bet_corr):
                fold_bet_corr = 0.0
        else:
            fold_bet_corr = 0.0

        # Entropy-like measure of street distribution variance
        streets_reached_arr = feature_arrays["streets_reached"]
        streets_entropy = float(np.std(streets_reached_arr))

        # Aggression consistency across hands (low std = bot-like)
        aggression_consistency = float(np.std(aggression_arr))

        # NEW: Check/call ratio correlation (behavioral homogeneity)
        if np.std(frac_check_arr) > 1e-9 and np.std(feature_arrays["frac_call"]) > 1e-9:
            check_call_corr = float(np.corrcoef(frac_check_arr, feature_arrays["frac_call"])[0, 1])
            if np.isnan(check_call_corr):
                check_call_corr = 0.0
        else:
            check_call_corr = 0.0

        # NEW: Pot-relative bet autocorrelation (bot sizing regularity)
        if len(pot_rel_bet_arr) > 1 and np.std(pot_rel_bet_arr) > 1e-9:
            pot_rel_bet_autocorr = float(np.corrcoef(pot_rel_bet_arr[:-1], pot_rel_bet_arr[1:])[0, 1])
            if np.isnan(pot_rel_bet_autocorr):
                pot_rel_bet_autocorr = 0.0
        else:
            pot_rel_bet_autocorr = 0.0

        # NEW: Aggression autocorrelation (hand-to-hand consistency)
        if len(aggression_arr) > 1 and np.std(aggression_arr) > 1e-9:
            aggr_autocorr = float(np.corrcoef(aggression_arr[:-1], aggression_arr[1:])[0, 1])
            if np.isnan(aggr_autocorr):
                aggr_autocorr = 0.0
        else:
            aggr_autocorr = 0.0

    else:
        bet_autocorr = 0.0
        pot_autocorr = 0.0
        preflop_aggr_corr = 0.0
        fold_bet_corr = 0.0
        streets_entropy = 0.0
        aggression_consistency = 0.0
        check_call_corr = 0.0
        pot_rel_bet_autocorr = 0.0
        aggr_autocorr = 0.0

    chunk_feats.extend([
        bet_autocorr, pot_autocorr, preflop_aggr_corr,
        fold_bet_corr, streets_entropy, aggression_consistency,
        check_call_corr, pot_rel_bet_autocorr, aggr_autocorr,
    ])

    # Percentile features for key distributions
    percentile_keys = [
        "frac_preflop", "stack_mean", "streets_reached",
        "aggression_ratio", "bet_mean", "final_pot", "frac_fold",
        "action_type_entropy", "pot_rel_bet_mean", "bet_cv",
        "aggression_street_std",
    ]
    for key in percentile_keys:
        arr = feature_arrays[key]
        p10 = float(np.percentile(arr, 10))
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        p90 = float(np.percentile(arr, 90))
        chunk_feats.extend([p10, p25, p75, p90, p75 - p25])  # IQR

    return np.array(chunk_feats, dtype=np.float32)


# Build feature names for the chunk-level feature vector
def _build_chunk_feature_names() -> List[str]:
    names = []
    for key in HAND_FEATURE_NAMES:
        names.append(f"{key}_mean")
        names.append(f"{key}_std")
        names.append(f"{key}_cv")
    names.append("n_hands")
    names.extend([
        "bet_autocorr", "pot_autocorr", "preflop_aggr_corr",
        "fold_bet_corr", "streets_entropy", "aggression_consistency",
        "check_call_corr", "pot_rel_bet_autocorr", "aggr_autocorr",
    ])
    percentile_keys = [
        "frac_preflop", "stack_mean", "streets_reached",
        "aggression_ratio", "bet_mean", "final_pot", "frac_fold",
        "action_type_entropy", "pot_rel_bet_mean", "bet_cv",
        "aggression_street_std",
    ]
    for key in percentile_keys:
        names.append(f"{key}_p10")
        names.append(f"{key}_p25")
        names.append(f"{key}_p75")
        names.append(f"{key}_p90")
        names.append(f"{key}_iqr")
    return names


CHUNK_FEATURE_NAMES = _build_chunk_feature_names()
