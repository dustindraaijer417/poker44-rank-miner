"""v24 Deeper Hierarchical Action Transformer scorer (gen2)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "model_v24_transformer.pkl"


def _bucket_amount(norm_bb: float, buckets: list) -> int:
    if norm_bb is None or norm_bb <= 0:
        return 0
    return min(range(len(buckets)), key=lambda i: abs(buckets[i] - norm_bb))


class V24Scorer:
    def __init__(self):
        import torch
        import torch.nn as nn
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"v24 transformer not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        self.embed_dim = int(payload["embed_dim"])
        self.n_heads = int(payload["n_heads"])
        self.n_layers_hand = int(payload["n_layers_hand"])
        self.n_layers_chunk = int(payload["n_layers_chunk"])
        self.ffn_dim = int(payload["ffn_dim"])
        self.action_types = payload["action_types"]
        self.streets = payload["streets"]
        self.amount_buckets = payload["amount_buckets"]
        self.max_actions = int(payload["max_actions_per_hand"])
        self.max_hands = int(payload["max_hands_per_chunk"])
        self.val_ap = float(payload.get("validation_ap", 0.0))

        EMBED_DIM = self.embed_dim
        N_HEADS = self.n_heads
        N_LAYERS_HAND = self.n_layers_hand
        N_LAYERS_CHUNK = self.n_layers_chunk
        FFN_DIM = self.ffn_dim
        N_ACT = len(self.action_types)
        N_STREET = len(self.streets)
        N_BUCKET = len(self.amount_buckets)
        MAX_A = self.max_actions
        MAX_H = self.max_hands

        class HierarchicalActionTransformerV22(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_emb = nn.Embedding(N_ACT, 32)
                self.street_emb = nn.Embedding(N_STREET, 12)
                self.seat_emb = nn.Embedding(10, 12)
                self.amount_emb = nn.Embedding(N_BUCKET, 32)
                self.action_proj = nn.Linear(32 + 12 + 12 + 32, EMBED_DIM)
                self.action_pos = nn.Parameter(torch.randn(MAX_A, EMBED_DIM) * 0.02)
                self.hand_cls = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
                self.chunk_cls = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
                hand_layer = nn.TransformerEncoderLayer(
                    d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
                    dropout=0.2, batch_first=True, norm_first=True,
                )
                self.hand_encoder = nn.TransformerEncoder(hand_layer, num_layers=N_LAYERS_HAND)
                self.hand_pos = nn.Parameter(torch.randn(MAX_H, EMBED_DIM) * 0.02)
                chunk_layer = nn.TransformerEncoderLayer(
                    d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
                    dropout=0.2, batch_first=True, norm_first=True,
                )
                self.chunk_encoder = nn.TransformerEncoder(chunk_layer, num_layers=N_LAYERS_CHUNK)
                self.head = nn.Sequential(
                    nn.LayerNorm(EMBED_DIM),
                    nn.Linear(EMBED_DIM, 96),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(96, 1),
                )

            def forward(self, x):
                B, H, A, _ = x.shape
                x_flat = x.view(B * H, A, 4)
                a, s, se, am = x_flat[..., 0], x_flat[..., 1], x_flat[..., 2], x_flat[..., 3]
                e = torch.cat([self.action_emb(a), self.street_emb(s),
                               self.seat_emb(se), self.amount_emb(am)], dim=-1)
                e = self.action_proj(e) + self.action_pos.unsqueeze(0)
                BH = B * H
                cls_h = self.hand_cls.expand(BH, 1, EMBED_DIM)
                e = torch.cat([cls_h, e], dim=1)
                action_mask = (a == N_ACT - 1)
                cls_mask = torch.zeros(BH, 1, dtype=torch.bool, device=action_mask.device)
                full_mask = torch.cat([cls_mask, action_mask], dim=1)
                e = self.hand_encoder(e, src_key_padding_mask=full_mask)
                hand_emb = e[:, 0, :].view(B, H, -1) + self.hand_pos.unsqueeze(0)
                chunk_cls = self.chunk_cls.expand(B, 1, EMBED_DIM)
                chunk_in = torch.cat([chunk_cls, hand_emb], dim=1)
                hand_mask = (x[:, :, 0, 0] == N_ACT - 1)
                cls_mask_chunk = torch.zeros(B, 1, dtype=torch.bool, device=hand_mask.device)
                full_chunk_mask = torch.cat([cls_mask_chunk, hand_mask], dim=1)
                chunk_e = self.chunk_encoder(chunk_in, src_key_padding_mask=full_chunk_mask)
                return self.head(chunk_e[:, 0, :]).squeeze(-1)

        self.model = HierarchicalActionTransformerV22()
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
            BATCH = 8
            probs = []
            for i in range(0, len(x_t), BATCH):
                logits = self.model(x_t[i:i + BATCH])
                probs.append(self.torch.sigmoid(logits).cpu().numpy())
            return np.clip(np.concatenate(probs), 0.0, 1.0)
