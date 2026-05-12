"""v20 — Hand-Action Transformer, an ensemble member with architectural diversity.

v17 and v19 are tree-based (XGB+LGBM) and rely on aggregated per-chunk features.
v20 is a different family: a small transformer that reads PER-ACTION sequences
within each hand and learns sequential patterns (call-call-call-fold vs
bet-raise-raise vs check-fold-check-fold) that flatten away when you compress
hands into per-chunk feature statistics.

Goal: add to the ensemble so the final score is (XGB + LGBM + transformer) / 3
(or weighted). Competitors can copy XGB+LGBM easily by reading our public code,
but a custom transformer trained on our pseudo-labeled live data is harder to
reproduce.

Architecture (small for fast training + small public weights):
  per-action: (action_type_id, street_id, actor_seat_id, amount_bucket_id) → 4 embeddings → concat → 32-dim
  positional encoding (sin-cos)
  2-layer TransformerEncoder (32-dim, 4 heads, ffn 64)
  mean pool over actions → hand embedding (32-dim)
  mean pool over hands → chunk embedding (32-dim)
  linear → bot logit

Trained on 4200 benchmark chunks (real GT) + ~70 pseudo-labeled live bots,
~2800 pseudo-labeled humans (semi-supervised, weighted lower).
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

DATA_DIR = PROJECT / "data" / "benchmark"
PSEUDO_PATH = PROJECT / "neurons" / "pseudo_live_labels.json"
CAP_DIR = PROJECT / "neurons" / "captured_chunks_h2"
MODEL_OUTPUT = PROJECT / "neurons" / "model_v20_transformer.pkl"

ACTION_TYPES = ["check", "call", "bet", "raise", "fold", "all_in", "small_blind", "big_blind", "other", ""]
STREETS = ["preflop", "flop", "turn", "river", ""]
MAX_ACTIONS_PER_HAND = 16
MAX_HANDS_PER_CHUNK = 100
AMOUNT_BUCKETS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0]
EMBED_DIM = 32


def _bucket_amount(norm_bb: float) -> int:
    if norm_bb is None or norm_bb <= 0:
        return 0
    nearest = min(range(len(AMOUNT_BUCKETS)), key=lambda i: abs(AMOUNT_BUCKETS[i] - norm_bb))
    return nearest


def encode_hand(hand: dict) -> np.ndarray:
    """(MAX_ACTIONS_PER_HAND, 4) int array of (action_type, street, seat, amount_bucket)."""
    encoded = np.zeros((MAX_ACTIONS_PER_HAND, 4), dtype=np.int64)
    actions = (hand or {}).get("actions", []) or []
    for i, a in enumerate(actions[:MAX_ACTIONS_PER_HAND]):
        at = (a.get("action_type") or "").lower().strip() or ""
        st = (a.get("street") or "").lower().strip() or ""
        seat = a.get("actor_seat", 0) or 0
        norm_bb = a.get("normalized_amount_bb", 0) or 0
        try:
            at_id = ACTION_TYPES.index(at)
        except ValueError:
            at_id = len(ACTION_TYPES) - 1
        try:
            st_id = STREETS.index(st)
        except ValueError:
            st_id = len(STREETS) - 1
        seat_id = max(0, min(int(seat), 9))
        amt_id = _bucket_amount(float(norm_bb))
        encoded[i] = (at_id, st_id, seat_id, amt_id)
    return encoded


def encode_chunk(chunk: list) -> np.ndarray:
    """(MAX_HANDS_PER_CHUNK, MAX_ACTIONS_PER_HAND, 4) int array."""
    encoded = np.zeros((MAX_HANDS_PER_CHUNK, MAX_ACTIONS_PER_HAND, 4), dtype=np.int64)
    if not chunk:
        return encoded
    for i, h in enumerate(chunk[:MAX_HANDS_PER_CHUNK]):
        encoded[i] = encode_hand(h)
    return encoded


def load_benchmark():
    cks, ys = [], []
    for f in sorted(DATA_DIR.glob("benchmark_*.json")):
        d = json.loads(f.read_text())
        for sc in d["data"]["chunks"]:
            sub = sc.get("chunks", [])
            gt = sc.get("groundTruth", [])
            for c, l in zip(sub, gt):
                cks.append(c); ys.append(int(l))
    return cks, np.array(ys, dtype=np.int64)


def load_pseudo():
    pseudo = json.loads(PSEUDO_PATH.read_text())
    by_file = {}
    for p in pseudo:
        by_file.setdefault(p["file"], []).append((p["idx"], p["label"]))
    cks, ys = [], []
    for fname, items in by_file.items():
        path = CAP_DIR / fname
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        for idx, lab in items:
            if idx < len(d.get("chunks", [])):
                cks.append(d["chunks"][idx]); ys.append(int(lab))
    return cks, np.array(ys, dtype=np.int64)


def main():
    import torch
    import torch.nn as nn

    print("=" * 60)
    print("Poker44 v20 Hand-Action Transformer")
    print("=" * 60)

    bench_chunks, bench_y = load_benchmark()
    pseudo_chunks, pseudo_y = load_pseudo()
    print(f"benchmark: {len(bench_chunks)} (bots={int(bench_y.sum())})")
    print(f"pseudo:    {len(pseudo_chunks)} (bots={int(pseudo_y.sum())})")

    print("\nEncoding chunks (token sequences)...")
    Xb = np.stack([encode_chunk(c) for c in bench_chunks])
    Xp = np.stack([encode_chunk(c) for c in pseudo_chunks])
    print(f"  bench tokens: {Xb.shape}")
    print(f"  pseudo tokens: {Xp.shape}")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, confusion_matrix

    X_tr, X_va, y_tr, y_va = train_test_split(
        Xb, bench_y, test_size=0.20, random_state=42, stratify=bench_y
    )
    sample_w_tr = np.ones(len(y_tr), dtype=np.float32)
    sample_w_p = np.full(len(pseudo_y), 0.3, dtype=np.float32)  # downweight pseudo
    X_all = np.concatenate([X_tr, Xp], axis=0)
    y_all = np.concatenate([y_tr, pseudo_y])
    w_all = np.concatenate([sample_w_tr, sample_w_p])
    print(f"\nCombined train set: {X_all.shape} bots={int(y_all.sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class HandActionTransformer(nn.Module):
        def __init__(self, embed_dim=EMBED_DIM):
            super().__init__()
            self.action_emb = nn.Embedding(len(ACTION_TYPES), 8)
            self.street_emb = nn.Embedding(len(STREETS), 4)
            self.seat_emb = nn.Embedding(10, 4)
            self.amount_emb = nn.Embedding(len(AMOUNT_BUCKETS), 8)
            self.proj = nn.Linear(8 + 4 + 4 + 8, embed_dim)
            self.pos_emb = nn.Parameter(torch.randn(MAX_ACTIONS_PER_HAND, embed_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=64,
                                                dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            # x: (B, H, A, 4) int64
            B, H, A, _ = x.shape
            x_flat = x.view(B * H, A, 4)
            a, s, se, am = x_flat[..., 0], x_flat[..., 1], x_flat[..., 2], x_flat[..., 3]
            e = torch.cat([self.action_emb(a), self.street_emb(s),
                           self.seat_emb(se), self.amount_emb(am)], dim=-1)
            e = self.proj(e) + self.pos_emb
            mask = (a == 0)  # actions with type_id=0 (check) and no other signal — keep all
            e = self.encoder(e)  # (B*H, A, D)
            hand_emb = e.mean(dim=1)  # (B*H, D)
            hand_emb = hand_emb.view(B, H, -1)  # (B, H, D)
            chunk_emb = hand_emb.mean(dim=1)  # (B, D)
            return self.head(chunk_emb).squeeze(-1)  # (B,)

    model = HandActionTransformer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    X_tr_t = torch.tensor(X_all, dtype=torch.long)
    y_tr_t = torch.tensor(y_all, dtype=torch.float32)
    w_tr_t = torch.tensor(w_all, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.long).to(device)
    y_va_np = y_va

    BATCH = 64
    N_EPOCH = 25
    n = len(X_tr_t)
    for ep in range(1, N_EPOCH + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = X_tr_t[idx].to(device)
            yb = y_tr_t[idx].to(device)
            wb = w_tr_t[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = (crit(logits, yb) * wb).mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / n
        model.eval()
        with torch.no_grad():
            p_va = []
            for i in range(0, len(X_va_t), BATCH):
                xb = X_va_t[i:i + BATCH]
                p_va.append(torch.sigmoid(model(xb)).cpu().numpy())
            p_va = np.concatenate(p_va)
        ap_va = average_precision_score(y_va_np, p_va)
        if ep == 1 or ep % 5 == 0 or ep == N_EPOCH:
            print(f"  epoch {ep:2d}  train_loss={train_loss:.4f}  bench_val_AP={ap_va:.4f}")

    preds = (p_va > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va_np, preds, labels=[0, 1]).ravel()
    print(f"\nFinal bench-val: AP={ap_va:.4f}  recall={tp/max(tp+fn,1):.4f}  fpr={fp/max(tn+fp,1):.4f}")

    # Save model
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "embed_dim": EMBED_DIM,
        "n_action_types": len(ACTION_TYPES),
        "n_streets": len(STREETS),
        "amount_buckets": AMOUNT_BUCKETS,
        "action_types": ACTION_TYPES,
        "streets": STREETS,
        "max_actions_per_hand": MAX_ACTIONS_PER_HAND,
        "max_hands_per_chunk": MAX_HANDS_PER_CHUNK,
        "validation_ap": float(ap_va),
        "version": 20,
    }
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved {MODEL_OUTPUT} ({MODEL_OUTPUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
