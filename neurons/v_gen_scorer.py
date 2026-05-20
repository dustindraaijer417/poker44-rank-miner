"""V_MLP scorer — simple MLP inspired by tomkaba's top-1 approach.
Raw numerical features, mean+max+min+std pooling, no attention.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "model_gen.pkl"

ACTION_TYPES = ["check", "call", "bet", "raise", "fold", "all_in", "small_blind", "big_blind", "other", ""]
STREETS = ["preflop", "flop", "turn", "river", ""]
N_NUM_FEATS = 12


def _safe(v, default=0.0):
    try:
        x = float(v) if v is not None else default
        return x if np.isfinite(x) else default
    except Exception:
        return default


class VGenScorer:
    def __init__(self):
        import torch
        import torch.nn as nn
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"v_gen model not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            p = pickle.load(f)
        self.max_hands = p["max_hands"]
        self.max_actions = p["max_actions"]
        self.action_types = p["action_types"]
        self.streets = p["streets"]
        AED = p["action_emb_dim"]
        SED = p["street_emb_dim"]
        SEED = p["seat_emb_dim"]
        NF = p["n_num_feats"]
        AH = p["action_hidden"]
        HR = p["hand_repr"]
        CH = p["chunk_hidden"]
        ACTION_INPUT_DIM = AED + SED + SEED + NF

        class V_MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_emb = nn.Embedding(len(ACTION_TYPES), AED)
                self.street_emb = nn.Embedding(len(STREETS), SED)
                self.seat_emb = nn.Embedding(10, SEED)
                self.action_mlp = nn.Sequential(
                    nn.Linear(ACTION_INPUT_DIM, AH),
                    nn.ReLU(),
                    nn.Linear(AH, HR),
                    nn.ReLU(),
                )
                self.chunk_mlp = nn.Sequential(
                    nn.Linear(HR * 4, CH),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(CH, 1),
                )

            def forward(self, cats, nums):
                B, H, A, _ = cats.shape
                action_e = self.action_emb(cats[..., 0])
                street_e = self.street_emb(cats[..., 1])
                seat_e = self.seat_emb(cats[..., 2])
                action_in = torch.cat([action_e, street_e, seat_e, nums], dim=-1)
                x = self.action_mlp(action_in)
                valid_mask = nums[..., 8].unsqueeze(-1)
                x = x * valid_mask
                denom = torch.clamp_min(valid_mask.sum(dim=2), 1.0)
                hand_repr = x.sum(dim=2) / denom
                hand_valid = (valid_mask.sum(dim=2) > 0).float()
                hand_repr = hand_repr * hand_valid
                h_denom = torch.clamp_min(hand_valid.sum(dim=1), 1.0)
                chunk_mean = hand_repr.sum(dim=1) / h_denom
                mask_neg = hand_repr + (1 - hand_valid) * (-1e9)
                mask_pos = hand_repr + (1 - hand_valid) * (1e9)
                chunk_max = mask_neg.max(dim=1).values
                chunk_min = mask_pos.min(dim=1).values
                diff = (hand_repr - chunk_mean.unsqueeze(1)) * hand_valid
                chunk_std = torch.sqrt(torch.clamp((diff * diff).sum(dim=1) / h_denom, min=1e-9))
                chunk_in = torch.cat([chunk_mean, chunk_max, chunk_min, chunk_std], dim=-1)
                return self.chunk_mlp(chunk_in).squeeze(-1)

        self.model = V_MLP()
        self.model.load_state_dict(p["state_dict"])
        self.model.eval()
        self.torch = torch
        self.val_ap = p.get("bench_val_ap", 0.0)

    def _encode_chunk(self, chunk):
        cat_ids = np.zeros((self.max_hands, self.max_actions, 3), dtype=np.int64)
        num_feats = np.zeros((self.max_hands, self.max_actions, N_NUM_FEATS), dtype=np.float32)
        if not chunk:
            return cat_ids, num_feats
        for h_idx, hand in enumerate(chunk[:self.max_hands]):
            if not hand:
                continue
            actions = (hand.get("actions") or [])[:self.max_actions]
            for a_idx, a in enumerate(actions):
                at = (a.get("action_type") or "").lower().strip() or ""
                st = (a.get("street") or "").lower().strip() or ""
                seat = a.get("actor_seat", 0) or 0
                try:
                    at_id = self.action_types.index(at)
                except ValueError:
                    at_id = len(self.action_types) - 1
                try:
                    st_id = self.streets.index(st)
                except ValueError:
                    st_id = len(self.streets) - 1
                seat_id = max(0, min(int(seat), 9))
                cat_ids[h_idx, a_idx] = (at_id, st_id, seat_id)
                amount = _safe(a.get("amount"))
                raise_to = a.get("raise_to")
                raise_to_missing = 0.0 if raise_to is not None else 1.0
                raise_to_v = _safe(raise_to)
                call_to = a.get("call_to")
                call_to_missing = 0.0 if call_to is not None else 1.0
                call_to_v = _safe(call_to)
                norm_bb = _safe(a.get("normalized_amount_bb"))
                pot_before = _safe(a.get("pot_before"))
                pot_after = _safe(a.get("pot_after"))
                valid_mask = 1.0
                num_feats[h_idx, a_idx] = [
                    amount, raise_to_v, call_to_v, norm_bb,
                    pot_before, pot_after, raise_to_missing, call_to_missing,
                    valid_mask, amount * valid_mask, pot_before * valid_mask, pot_after * valid_mask,
                ]
        return cat_ids, num_feats

    def score_batch(self, chunks: List) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        cats_list, nums_list = [], []
        for c in chunks:
            ci, ni = self._encode_chunk(c)
            cats_list.append(ci); nums_list.append(ni)
        cats = np.stack(cats_list)
        nums = np.stack(nums_list)
        c_t = self.torch.tensor(cats, dtype=self.torch.long)
        n_t = self.torch.tensor(nums, dtype=self.torch.float32)
        with self.torch.no_grad():
            BATCH = 32
            probs = []
            for i in range(0, len(c_t), BATCH):
                logits = self.model(c_t[i:i + BATCH], n_t[i:i + BATCH])
                probs.append(self.torch.sigmoid(logits).cpu().numpy())
            return np.clip(np.concatenate(probs), 0.0, 1.0)
