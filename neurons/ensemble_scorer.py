"""Blended scorer combining v19 (XGB+LGBM stable features) with v21 transformer.

v21 is a 67K-param hierarchical action transformer trained on benchmark
labels + auxiliary human corpus (poker_hands_combined.json.gz, 32K hands).
The transformer captures action-sequence patterns that v19's aggregated
features cannot, and the aux corpus gives us a strong human baseline.

Weights are configurable per hotkey:
- weight_v19=1.0 -> pure v19 (conservative, ~1% bot calls on live)
- weight_v19=0.5 -> balanced (~4-30% depending on agreement)
- weight_v19=0.0 -> pure v21 (aggressive, ~32% bot calls on live)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


class EnsembleScorer:
    def __init__(self, weight_v19: float = 0.5):
        from neurons.v19_scorer import V19Scorer
        self.v19 = V19Scorer()
        self.weight_v19 = float(weight_v19)
        self.v21: Optional[object] = None
        try:
            from neurons.v21_scorer import V21Scorer
            v21_path = Path(__file__).resolve().parent / "model_v21_transformer.pkl"
            if v21_path.exists():
                self.v21 = V21Scorer()
        except Exception:
            self.v21 = None

    def score_batch(self, chunks: List) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        p19 = self.v19.score_batch(chunks)
        if self.v21 is None or self.weight_v19 >= 1.0:
            return p19
        p21 = self.v21.score_batch(chunks)
        if self.weight_v19 <= 0.0:
            return np.clip(p21, 0.0, 1.0)
        return np.clip(self.weight_v19 * p19 + (1.0 - self.weight_v19) * p21, 0.0, 1.0)
