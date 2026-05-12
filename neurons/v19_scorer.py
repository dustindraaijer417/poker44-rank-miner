"""v19 scorer — XGB + LGBM ensemble on 118 stable features.

Drop-in replacement for V17Scorer. v19 was trained on the same 4200
labeled benchmark chunks as v17, but restricted to features whose
distribution is stable between benchmark and live data (identified by
analyze_feature_stability.py).

Head-to-head on 4000 live captured chunks: v19 detects 4.1% bots vs
v17's 2.7%, keeps all of v17's high-confidence calls, and resolves
~20% of v17's uncertain zone into decisive predictions with zero new
false positives on v17's confident humans.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from neurons.v14_features import extract_v14_features

MODEL_PATH = Path(__file__).resolve().parent / "model_v19.pkl"


class V19Scorer:
    """Loads v19 (stable-features ensemble) and scores chunks."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"v19 model not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        self.xgb = payload["xgb_model"]
        self.lgbm = payload["lgbm_model"]
        self.weight_xgb = float(payload.get("ensemble_weight_xgb", 0.2))
        self.feature_names = payload["feature_names"]
        self.stable_indices = payload["stable_indices"]
        self.val_ap = float(payload.get("validation_ap", 1.0))

    def score_batch(self, chunks: List) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        X = []
        for c in chunks:
            feat = extract_v14_features(c) if c else np.zeros(355, dtype=np.float32)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            X.append(feat)
        X = np.array(X, dtype=np.float32)[:, self.stable_indices]
        df = pd.DataFrame(X, columns=self.feature_names)
        p_xgb = np.clip(self.xgb.predict_proba(df.values)[:, 1], 0.0, 1.0)
        p_lgbm = np.clip(self.lgbm.predict_proba(df.values)[:, 1], 0.0, 1.0)
        return self.weight_xgb * p_xgb + (1 - self.weight_xgb) * p_lgbm
