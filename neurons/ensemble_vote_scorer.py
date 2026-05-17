"""Multi-model voting ensemble scorer (v19+v21+v22+v24).

For each chunk, score with all 4 models then aggregate:
- Strong bot consensus (3+ vote bot AND avg > 0.6): output 0.92
- Strong human consensus (3+ vote human AND avg < 0.1): output 0.05
- Otherwise: weighted average, leaning conservative

This gives high-confidence predictions only where multiple architectures
agree, reducing FPR risk while maintaining recall on clear cases.
"""
from __future__ import annotations

from typing import List

import numpy as np


class EnsembleVoteScorer:
    def __init__(self):
        from neurons.v19_scorer import V19Scorer
        from neurons.v21_scorer import V21Scorer
        from neurons.v22_scorer import V22Scorer
        from neurons.v24_scorer import V24Scorer
        self.v19 = V19Scorer()
        self.v21 = V21Scorer()
        self.v22 = V22Scorer()
        self.v24 = V24Scorer()

    def score_batch(self, chunks: List) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        s19 = self.v19.score_batch(chunks)
        s21 = self.v21.score_batch(chunks)
        s22 = self.v22.score_batch(chunks)
        s24 = self.v24.score_batch(chunks)

        bot_votes = (s19 > 0.5).astype(int) + (s21 > 0.5).astype(int) + (s22 > 0.5).astype(int) + (s24 > 0.5).astype(int)
        human_votes = (s19 < 0.05).astype(int) + (s21 < 0.05).astype(int) + (s22 < 0.05).astype(int) + (s24 < 0.05).astype(int)
        avg = (s19 + s21 + s22 + s24) / 4.0

        out = np.zeros(len(chunks), dtype=float)
        # Tiered confidence:
        strong_bot = (bot_votes >= 3) & (avg > 0.5)
        medium_bot = (bot_votes >= 2) & (avg > 0.3) & ~strong_bot
        strong_human = (human_votes >= 3) & (avg < 0.1)
        # Default: damped average
        out[:] = avg * 0.7
        out[strong_bot] = np.clip(0.85 + 0.1 * avg[strong_bot], 0.5, 0.97)
        out[medium_bot] = np.clip(0.55 + 0.2 * avg[medium_bot], 0.5, 0.85)
        out[strong_human] = np.clip(0.02 + 0.5 * avg[strong_human], 0.0, 0.1)
        return np.clip(out, 0.0, 1.0)
