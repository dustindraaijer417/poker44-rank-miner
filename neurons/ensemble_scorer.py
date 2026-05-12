"""Blended scorer combining v19 (XGB+LGBM stable features) with v20 transformer.

The ensemble exists so a single architecture failure doesn't blank our signal:
- v19 is a tree model on aggregated structural features — robust to most
  payload changes, fast inference, easy to inspect.
- v20 is a small transformer over per-action sequences — captures patterns
  that aggregation throws away (call-call-call-fold vs bet-raise-raise).

Drop-in replacement for V19Scorer in the miner; if v20 model file is missing
or fails to load, falls back to v19 alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


class EnsembleScorer:
    """v19 + v20 weighted ensemble. weight_v19 in [0, 1], v20 gets (1 - weight_v19)."""

    def __init__(self, weight_v19: float = 0.5):
        from neurons.v19_scorer import V19Scorer
        self.v19 = V19Scorer()
        self.weight_v19 = float(weight_v19)
        self.v20: Optional[object] = None
        try:
            from neurons.v20_scorer import V20Scorer
            v20_path = Path(__file__).resolve().parent / "model_v20_transformer.pkl"
            if v20_path.exists():
                self.v20 = V20Scorer()
        except Exception:
            self.v20 = None

    def score_batch(self, chunks: List) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        p19 = self.v19.score_batch(chunks)
        if self.v20 is None:
            return p19
        p20 = self.v20.score_batch(chunks)
        return np.clip(self.weight_v19 * p19 + (1.0 - self.weight_v19) * p20, 0.0, 1.0)
