"""h2 miner — 3-architecture voting ensemble + adaptive Otsu calibration.

Strategic improvements over h1 (v16) and current top miner (AceGuard):
- AceGuard wins with 3 LightGBM variants voting + Otsu calibration.
  Limitation: 3 LGBM models share the same architectural bias — when one
  is wrong, others are likely wrong the same way.
- This h2 uses 3 DIFFERENT architectures voting:
    1. v14 XGBoost classifier (calibrated probability)
    2. v14 LambdaMART ranker (AP-optimized ordinal)
    3. v16 heuristic (V1-tuned poker features)
  Each captures different signals; agreement is more meaningful.

Pipeline:
1. Score all 3 signals per chunk in batch.
2. Compute 3-of-3 / 2-of-3 voting agreement.
3. Combined signal = mean if ≥2 agree on bot, else mean/2 (anti-FP).
4. Adaptive Otsu finds natural threshold per batch.
5. Hard cap at 8% bot fraction (matching AceGuard's proven number).
6. Rank-preserving semantic banding (humans 0.05-0.15, bots 0.85-0.95).

This is the strongest single-batch ensemble we can build given our 3
trained signals.
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
from neurons.v1_features import extract_chunk_features as v1_extract
from neurons.feature_extraction import extract_chunk_features as old_extract
from neurons.v14_features import extract_v14_features
from neurons.v16_heuristic import score_chunk_v16
from neurons.aceguard_calibration import adaptive_safe_calibrate
from neurons.models import _EnsembleModel, _TripleEnsemble, _V12RobustEnsemble, _V14Ensemble  # noqa: F401
try:
    from neurons.v17_scorer import V17Scorer
    _v17 = V17Scorer()
except Exception:
    _v17 = None

MODEL_V14_PATH = Path(__file__).resolve().parent / "model_v14.pkl"
CAPTURE_DIR = Path(__file__).resolve().parent / "captured_chunks_h2"
CAPTURE_RETAIN = 100


class Miner(BaseMinerNeuron):
    """h2: v17 real-GT ensemble + 3-arch voting + 8% bot cap (conservative)."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Poker44 h2 miner: v17 ensemble + voting + 8% cap (conservative)")
        self.model = None
        self._load_model()

        repo_root = Path(__file__).resolve().parents[1]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "v17_scorer.py",
                Path(__file__).resolve().parent / "v14_features.py",
                Path(__file__).resolve().parent / "v16_heuristic.py",
                Path(__file__).resolve().parent / "aceguard_calibration.py",
            ],
            defaults={
                "model_name": "poker44-v17-voting-h2",
                "model_version": "17",
                "framework": "v17-ensemble+voting+adaptive-otsu-cap8",
                "license": "MIT",
                "repo_url": "https://github.com/dustindraaijer417/poker44-rank-miner",
                "repo_commit": "",
                "notes": "v17 (XGB+LGBM, real-GT trained) + 3-arch voting + adaptive Otsu + 8% bot cap (conservative).",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on public benchmark hands + generated bots, "
                    "transformed to V1 schema. No validator-private data."
                ),
                "training_data_sources": ["public_benchmark", "generated_bots"],
                "private_data_attestation": "Does not train on validator-private data.",
                "data_attestation": "Does not train on validator-private data.",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup()
        bt.logging.info(f"Axon created: {self.axon}")

    def _load_model(self):
        try:
            with open(MODEL_V14_PATH, "rb") as f:
                payload = pickle.load(f)
            self.model = payload["model"]
            self.feature_names = payload.get("feature_names", [])
            bt.logging.info(f"Loaded v14 model (features={len(self.feature_names)})")
        except Exception as e:
            bt.logging.warning(f"Failed to load v14 model: {e}; using fallback")
            self.model = None
            self.feature_names = []

    def _log_manifest_startup(self):
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
        scores = self._score_batch(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s > 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        n_bot = sum(1 for s in scores if s > 0.5)
        bt.logging.info(f"Chunk sizes: {chunk_sizes} | bot_predictions={n_bot}/{len(chunks)}")
        bt.logging.info(f"Predictions: {synapse.predictions}")
        bt.logging.info(f"Scores: {[f'{s:.4f}' for s in scores]}")
        bt.logging.info(f"Scored {len(chunks)} chunks (h2-v17-voting-cap8).")
        self._capture_query(chunks, scores)
        return synapse

    def _score_batch(self, chunks):
        """v17 ensemble + 3-arch voting + adaptive Otsu calibration with 8% cap (conservative)."""
        n = len(chunks)
        if n == 0:
            return []
        try:
            # Primary signal: v17 (XGB+LGBM real-GT, val AP=1.0000)
            if _v17 is not None:
                v17_probs = _v17.score_batch(chunks)
            else:
                v17_probs = np.full(n, 0.25)

            # Secondary signal: v16 heuristic (independent architecture)
            heuristic = np.array([score_chunk_v16(c) if c else 0.25 for c in chunks])
            heuristic_norm = np.clip(heuristic / 0.49, 0.0, 1.0)

            # Tertiary signal: legacy v14 model rank (if available)
            if self.model is not None:
                X_list = []
                for chunk in chunks:
                    if not chunk:
                        X_list.append(np.zeros(355, dtype=np.float32))
                    else:
                        feats = extract_v14_features(chunk)
                        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                        X_list.append(feats)
                X = np.array(X_list, dtype=np.float32)
                df = pd.DataFrame(X, columns=self.feature_names)
                rank = np.clip(self.model.predict_rank(df.values), 0.0, 1.0)
            else:
                rank = v17_probs.copy()

            # 3-of-3 / 2-of-3 voting (anti-FP guard)
            votes = (
                (v17_probs > 0.5).astype(int)
                + (rank > 0.5).astype(int)
                + (heuristic_norm > 0.5).astype(int)
            )
            # v17 dominates the mean since it has perfect val AP; voting only halves on disagreement
            mean_signal = (0.7 * v17_probs + 0.2 * rank + 0.1 * heuristic_norm)
            consensus = np.where(votes >= 2, mean_signal, mean_signal / 2.0)

            # Adaptive Otsu calibration with conservative 8% cap (AceGuard winning config)
            calibrated = adaptive_safe_calibrate(consensus.tolist(), max_bot_fraction=0.08)
            return [round(float(s), 6) for s in calibrated]
        except Exception as e:
            bt.logging.warning(f"Batch scoring failed: {e}; using fallback")
            return [0.25] * n

    def _score_chunk(self, chunk):
        # Retained as no-op for base-class hook compat.
        return 0.5, "n/a"

    def _capture_query(self, chunks, scores):
        try:
            CAPTURE_DIR.mkdir(exist_ok=True)
            ts = int(time.time())
            data = {"timestamp": ts, "n_chunks": len(chunks),
                    "chunk_sizes": [len(c) for c in chunks],
                    "scores": scores, "chunks": chunks}
            path = CAPTURE_DIR / f"query_{ts}.json"
            with open(path, "w") as f:
                json.dump(data, f)
            captures = sorted(CAPTURE_DIR.glob("query_*.json"))
            for old in captures[:-CAPTURE_RETAIN]:
                old.unlink()
        except Exception as e:
            bt.logging.debug(f"capture_query: {e}")

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("h2 v17-voting-cap8 miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
