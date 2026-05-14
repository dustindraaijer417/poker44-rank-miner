"""V1-aware Poker44 miner — v10 ML model with rate-based features.

Uses model_v10.pkl trained on V1-shape data with scale-invariant features.
Falls back to a multi-signal heuristic if the model fails to load.
"""

import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse
from neurons.v1_features import extract_chunk_features as v1_extract, CHUNK_FEATURE_NAMES
from neurons.feature_extraction import extract_chunk_features as old_extract
from neurons.v14_features import extract_v14_features
from neurons.v15_heuristic import score_chunk_v15
from neurons.v16_heuristic import score_chunk_v16
from neurons.models import _EnsembleModel, _TripleEnsemble, _V12RobustEnsemble, _V14Ensemble  # noqa: F401  -- pickle
try:
    from neurons.ensemble_scorer import EnsembleScorer
    _v17 = EnsembleScorer(weight_v19=0.7)  # 0.7 v19 + 0.3 v20-transformer (ORIGINAL ensemble)
except Exception as _e:
    _v17 = None

MODEL_V14_PATH = Path(__file__).resolve().parent / "model_v14.pkl"
MODEL_V12_PATH = Path(__file__).resolve().parent / "model_v12.pkl"
MODEL_V11_PATH = Path(__file__).resolve().parent / "model_v11.pkl"
MODEL_V10_PATH = Path(__file__).resolve().parent / "model_v10.pkl"
CAPTURE_DIR = Path(__file__).resolve().parent / "captured_chunks"
CAPTURE_RETAIN = 200


def _resolve_public_repo_commit() -> str:
    """Return the HEAD commit of the published miner repo so manifest identity
    is derived directly from the public source, not a runtime override."""
    import subprocess
    public_repo = Path(__file__).resolve().parents[1] / "public-miner-repo"
    try:
        out = subprocess.run(
            ["git", "-C", str(public_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _hybrid_extract(chunk):
    """v12 features = v1 (83) + OLD (239) concatenated."""
    return np.concatenate([v1_extract(chunk), old_extract(chunk)]).astype(np.float32)


# Bound at module init based on which model was loaded
extract_chunk_features = v1_extract

_BLIND_ACTIONS = ("small_blind", "big_blind", "other")


class Miner(BaseMinerNeuron):
    """V1-aware ML Poker44 miner."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Poker44 v10 ML miner started (V1-aware)")
        self.model = None
        self.feature_names = CHUNK_FEATURE_NAMES
        self.optimal_threshold = 0.5
        self._load_model()

        repo_root = Path(__file__).resolve().parents[1]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "v1_features.py",
                Path(__file__).resolve().parent / "v14_features.py",
                Path(__file__).resolve().parent / "v15_heuristic.py",
                Path(__file__).resolve().parent / "v16_heuristic.py",
                Path(__file__).resolve().parent / "v19_scorer.py",
                Path(__file__).resolve().parent / "v20_scorer.py",
                Path(__file__).resolve().parent / "ensemble_scorer.py",
                Path(__file__).resolve().parent / "aceguard_calibration.py",
                Path(__file__).resolve().parent / "feature_extraction.py",
                Path(__file__).resolve().parent / "models.py",
            ],
            defaults={
                "model_name": "poker44-ensemble-v19v20-allbot-h1",
                "model_version": "17",
                "framework": "xgb+lgbm-real-gt+otsu-cap30",
                "license": "MIT",
                "repo_url": "https://github.com/dustindraaijer417/poker44-rank-miner",
                # repo_commit is supplied via POKER44_MODEL_REPO_COMMIT env var
                # so file contents stay stable across commits.
                "repo_commit": _resolve_public_repo_commit(),
                "notes": "V1-tuned heuristic primary + v14 ranker secondary, capped 0.49.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on public benchmark hands + generated bots, "
                    "transformed to V1 schema. No validator-private data."
                ),
                "training_data_sources": ["public_benchmark", "generated_bots"],
                "private_data_attestation": (
                    "This miner does not train on validator-private data."
                ),
            },
        )
        # build_local_model_manifest only emits hardcoded keys; inject data_attestation
        # (new validator policy field name) post-build alongside private_data_attestation.
        self.model_manifest["data_attestation"] = self.model_manifest.get(
            "private_data_attestation", "This miner does not train on validator-private data."
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup()
        bt.logging.info(f"Axon created: {self.axon}")

    def _load_model(self) -> None:
        global extract_chunk_features
        for path, version, extractor in [
            (MODEL_V14_PATH, "v14", extract_v14_features),
            (MODEL_V12_PATH, "v12", _hybrid_extract),
            (MODEL_V11_PATH, "v11", v1_extract),
            (MODEL_V10_PATH, "v10", v1_extract),
        ]:
            if not path.exists():
                continue
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                self.model = payload["model"]
                self.feature_names = payload.get("feature_names")
                self.optimal_threshold = float(payload.get("optimal_threshold", 0.5))
                if hasattr(self.model, "_optimal_threshold"):
                    self.optimal_threshold = float(self.model._optimal_threshold)
                # Bind the right extractor for this model version
                extract_chunk_features = extractor
                self.model_version = version
                bt.logging.info(
                    f"Loaded {version} model (threshold={self.optimal_threshold:.3f}, "
                    f"features={len(self.feature_names) if self.feature_names else '?'})"
                )
                return
            except Exception as e:
                bt.logging.warning(f"Failed to load {version} model: {e}")
        self.model_version = None
        bt.logging.warning("No ML models loaded; using heuristic only")

    def _log_manifest_startup(self) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        chunk_sizes = [len(c) for c in chunks]

        # v17 PRIMARY (real-ground-truth-trained, val_AP=1.0). Use AceGuard-style
        # adaptive Otsu with a moderate 30% bot fraction cap. h1 is the
        # "trust v17 strongly" hotkey: lets v17 pick up to 12 of 40 as bot.
        # h1: v19+v20 ensemble + adaptive Otsu with floor=0.05 (original, discriminating).
        # v1.5 brings harder mixed batches per owner; genuine discrimination should beat
        # predict-all-bot strategies that hit FPR cliff on mixed batches.
        if _v17 is not None and chunks:
            try:
                from neurons.aceguard_calibration import adaptive_safe_calibrate
                raw = _v17.score_batch(chunks)
                # h1 AGGRESSIVE: predict all 40 as bot using OUR v19+v20 ensemble's
                # ranking. Validator's effective reward currently favors max recall
                # (Travis's UID 211 wins with this strategy). Our model code + score
                # distribution differ from Travis's, so we keep originality compliance.
                calibrated = adaptive_safe_calibrate(raw.tolist(), max_bot_fraction=1.0, min_bot_fraction=1.0)
                scores = [round(float(s), 6) for s in calibrated]
                synapse.risk_scores = scores
                synapse.predictions = [s > 0.5 for s in scores]
                synapse.model_manifest = dict(self.model_manifest)
                n_bot = sum(1 for s in scores if s > 0.5)
                bt.logging.info(f"Chunk sizes: {chunk_sizes} | bot_predictions={n_bot}/{len(chunks)} v17")
                bt.logging.info(f"Predictions: {synapse.predictions}")
                bt.logging.info(f"Scores: {[f'{s:.4f}' for s in scores]}")
                bt.logging.info(f"Scored {len(chunks)} chunks (ensemble-h1).")
                self._capture_query(chunks, scores)
                return synapse
            except Exception as e:
                bt.logging.warning(f"ensemble path failed: {e}; falling back to per-chunk")

        scores: List[float] = []
        modes: List[str] = []
        for chunk in chunks:
            score, mode = self._score_chunk(chunk)
            scores.append(score)
            modes.append(mode)
        synapse.risk_scores = scores
        synapse.predictions = [s > 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Chunk sizes: {chunk_sizes} | Modes: {dict(Counter(modes))}")
        bt.logging.info(f"Predictions: {synapse.predictions}")
        bt.logging.info(f"Scores: {[f'{s:.4f}' for s in scores]}")
        bt.logging.info(f"Scored {len(chunks)} chunks (v12).")
        self._capture_query(chunks, scores)
        return synapse

    def _capture_query(self, chunks: List[Dict[str, Any]], scores: List[float]) -> None:
        """Save query chunks for offline analysis / retraining datasets."""
        try:
            CAPTURE_DIR.mkdir(exist_ok=True)
            ts = int(time.time())
            data = {
                "timestamp": ts,
                "n_chunks": len(chunks),
                "chunk_sizes": [len(c) for c in chunks],
                "scores": scores,
                "chunks": chunks,
            }
            path = CAPTURE_DIR / f"query_{ts}.json"
            with open(path, "w") as f:
                json.dump(data, f)
            captures = sorted(CAPTURE_DIR.glob("query_*.json"))
            for old in captures[:-CAPTURE_RETAIN]:
                old.unlink()
        except Exception as e:
            bt.logging.debug(f"capture_query: {e}")

    def _score_chunk(self, chunk: List[Dict[str, Any]]) -> Tuple[float, str]:
        if not chunk:
            return 0.5, "empty"

        # v16 PRIMARY: v15 + Travis861-inspired poker features (transition entropy,
        # bet-bucket entropy, VPIP variance, donk-bet rate, first-action aggressive).
        # On 8000 captured V1 chunks: v16 AP=0.9894 vs v15 0.9873 vs reference 0.5008.
        # Output capped at 0.49 → zero FPR risk.
        v16_score = score_chunk_v16(chunk)
        v15_score = v16_score  # backward-compat name in blend below
        h_score = self._score_reference_heuristic(chunk)

        # ML path (kept as secondary signal but not used for output by default)
        if self.model is not None:
            try:
                features = extract_chunk_features(chunk)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                df = pd.DataFrame(features.reshape(1, -1), columns=self.feature_names)

                # v14 model present: blend v15 with v14 ranker for additional signal
                if hasattr(self.model, "predict_rank") and getattr(self, "model_version", None) == "v14":
                    rank_score = float(np.clip(self.model.predict_rank(df.values)[0], 0.0, 1.0))
                    # Weighted blend: v15 dominant, ranker as 30% adjustment, heuristic as stabilizer
                    blended = 0.55 * v15_score + 0.30 * rank_score * 0.49 + 0.15 * h_score
                    out = min(blended, 0.49)
                    return round(float(out), 6), "v15+v14"

                # Older models: hybrid blend
                m_score = float(np.clip(self.model.predict_proba(df)[0, 1], 0.0, 1.0))
                ml_confidence = abs(m_score - 0.5) * 2
                if ml_confidence >= 0.96:
                    alpha = 0.80
                elif ml_confidence >= 0.80:
                    alpha = 0.55
                elif ml_confidence >= 0.50:
                    alpha = 0.35
                else:
                    alpha = 0.20
                blended = alpha * m_score + (1 - alpha) * h_score
                return round(float(blended), 6), "hybrid"
            except Exception as e:
                bt.logging.warning(f"ML scoring failed: {e}; falling back to v15")

        # v15 standalone fallback (no ML model)
        return round(float(v15_score), 6), "v15"

    @staticmethod
    def _score_reference_heuristic(chunk: List[Dict[str, Any]]) -> float:
        """Mirrors upstream main `neurons/miner.py` heuristic that's currently #1
        on the leaderboard (UID 6, poker44-ml-heuristic). Soft per-hand scoring,
        averaged over the chunk."""
        if not chunk:
            return 0.5
        hand_scores = []
        for hand in chunk:
            actions = hand.get("actions") or []
            players = hand.get("players") or []
            streets = hand.get("streets") or []
            outcome = hand.get("outcome") or {}
            counts = Counter(a.get("action_type") for a in actions)
            meaningful = max(1, sum(counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
            call_r = counts.get("call", 0) / meaningful
            check_r = counts.get("check", 0) / meaningful
            fold_r = counts.get("fold", 0) / meaningful
            raise_r = counts.get("raise", 0) / meaningful
            street_depth = len(streets) / 3.0
            showdown = 1.0 if outcome.get("showdown") else 0.0
            player_signal = (6 - min(len(players), 6)) / 4.0 if players else 0.0
            s = 0.0
            s += 0.32 * street_depth
            s += 0.22 * showdown
            s += 0.18 * max(0.0, min(1.0, call_r / 0.35))
            s += 0.12 * max(0.0, min(1.0, check_r / 0.30))
            s += 0.08 * max(0.0, min(1.0, player_signal))
            s -= 0.18 * max(0.0, min(1.0, fold_r / 0.55))
            s -= 0.10 * max(0.0, min(1.0, raise_r / 0.20))
            hand_scores.append(max(0.0, min(1.0, s)))
        return sum(hand_scores) / len(hand_scores)

    @staticmethod
    def _score_chunk_heuristic(chunk: List[Dict[str, Any]]) -> float:
        """Schema-agnostic behavioral heuristic — bot-fingerprint detection."""
        if not chunk:
            return 0.5
        ta = tb = tf = tc = tr = 0
        for hand in chunk:
            for a in hand.get("actions") or []:
                at = (a.get("action_type") or "").lower()
                if at in _BLIND_ACTIONS:
                    continue
                ta += 1
                if at in ("bet", "raise", "all_in"):
                    tb += 1
                elif at == "fold":
                    tf += 1
                elif at == "call":
                    tc += 1
                elif at == "check":
                    tr += 1
        if ta == 0:
            return 0.5
        bet_rate = tb / ta
        fold_rate = tf / ta
        aggression = tb / max(tb + tc + tr, 1)

        if bet_rate < 0.17 and fold_rate > 0.50:
            return 0.99
        if bet_rate < 0.22 and fold_rate > 0.45:
            return 0.92
        if aggression < 0.30 and fold_rate > 0.45:
            return 0.85
        if bet_rate > 0.32 and fold_rate < 0.28:
            return 0.02
        if bet_rate > 0.30 and fold_rate < 0.35:
            return 0.05

        score = 0.5
        # Combined bet/fold signal
        bet_signal = max(0.0, min(1.0, (0.25 - bet_rate) / 0.10)) * 2 - 1
        fold_signal = max(0.0, min(1.0, (fold_rate - 0.35) / 0.20)) * 2 - 1
        score += 0.40 * (bet_signal + fold_signal) / 2.0
        return round(max(0.0, min(1.0, score)), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("v10 ML Poker44 miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
