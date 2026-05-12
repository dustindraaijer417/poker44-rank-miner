"""v20 Hand-Action Transformer scorer.

Loads the trained transformer state_dict and scores chunks via sequence-level
modeling of per-hand action patterns. Designed as an ensemble member alongside
V19Scorer (XGB+LGBM on stable features); the ensemble decision can be:

    p_final = 0.5 * v19.score_batch(chunks) + 0.5 * v20.score_batch(chunks)

(weight tunable per miner — h1/h2/h3 can pick different blends).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "model_v20_transformer.pkl"


def _bucket_amount(norm_bb: float, buckets: list) -> int:
    if norm_bb is None or norm_bb <= 0:
        return 0
    return min(range(len(buckets)), key=lambda i: abs(buckets[i] - norm_bb))


class V20Scorer:
    """Loads transformer state_dict + scores chunks. CPU-only is fine (small model)."""

    def __init__(self):
        import torch
        import torch.nn as nn
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"v20 transformer not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        self.embed_dim = int(payload["embed_dim"])
        self.action_types = payload["action_types"]
        self.streets = payload["streets"]
        self.amount_buckets = payload["amount_buckets"]
        self.max_actions = int(payload["max_actions_per_hand"])
        self.max_hands = int(payload["max_hands_per_chunk"])
        self.val_ap = float(payload.get("validation_ap", 1.0))

        # Rebuild the same architecture used in training
        class HandActionTransformer(nn.Module):
            def __init__(self, embed_dim, n_action_types, n_streets, n_buckets,
                          max_actions):
                super().__init__()
                self.action_emb = nn.Embedding(n_action_types, 8)
                self.street_emb = nn.Embedding(n_streets, 4)
                self.seat_emb = nn.Embedding(10, 4)
                self.amount_emb = nn.Embedding(n_buckets, 8)
                self.proj = nn.Linear(8 + 4 + 4 + 8, embed_dim)
                self.pos_emb = nn.Parameter(torch.randn(max_actions, embed_dim) * 0.02)
                layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                                    dim_feedforward=64, dropout=0.1,
                                                    batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(self, x):
                B, H, A, _ = x.shape
                x_flat = x.view(B * H, A, 4)
                a, s, se, am = x_flat[..., 0], x_flat[..., 1], x_flat[..., 2], x_flat[..., 3]
                e = torch.cat([self.action_emb(a), self.street_emb(s),
                                self.seat_emb(se), self.amount_emb(am)], dim=-1)
                e = self.proj(e) + self.pos_emb
                e = self.encoder(e)
                hand_emb = e.mean(dim=1).view(B, H, -1)
                chunk_emb = hand_emb.mean(dim=1)
                return self.head(chunk_emb).squeeze(-1)

        self.model = HandActionTransformer(
            embed_dim=self.embed_dim,
            n_action_types=len(self.action_types),
            n_streets=len(self.streets),
            n_buckets=len(self.amount_buckets),
            max_actions=self.max_actions,
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.torch = torch

    def _encode_hand(self, hand: dict) -> np.ndarray:
        encoded = np.zeros((self.max_actions, 4), dtype=np.int64)
        actions = (hand or {}).get("actions", []) or []
        for i, a in enumerate(actions[:self.max_actions]):
            at = (a.get("action_type") or "").lower().strip() or ""
            st = (a.get("street") or "").lower().strip() or ""
            seat = a.get("actor_seat", 0) or 0
            norm_bb = a.get("normalized_amount_bb", 0) or 0
            try:
                at_id = self.action_types.index(at)
            except ValueError:
                at_id = len(self.action_types) - 1
            try:
                st_id = self.streets.index(st)
            except ValueError:
                st_id = len(self.streets) - 1
            seat_id = max(0, min(int(seat), 9))
            amt_id = _bucket_amount(float(norm_bb), self.amount_buckets)
            encoded[i] = (at_id, st_id, seat_id, amt_id)
        return encoded

    def _encode_chunk(self, chunk: list) -> np.ndarray:
        encoded = np.zeros((self.max_hands, self.max_actions, 4), dtype=np.int64)
        if not chunk:
            return encoded
        for i, h in enumerate(chunk[:self.max_hands]):
            encoded[i] = self._encode_hand(h)
        return encoded

    def score_batch(self, chunks: List) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        X = np.stack([self._encode_chunk(c) for c in chunks])
        x_t = self.torch.tensor(X, dtype=self.torch.long)
        with self.torch.no_grad():
            logits = self.model(x_t)
            probs = self.torch.sigmoid(logits).cpu().numpy()
        return np.clip(probs, 0.0, 1.0)
