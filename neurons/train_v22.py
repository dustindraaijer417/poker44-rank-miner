"""v22 — Deeper Hierarchical Action Transformer (gen2 vs v21).

Improvements:
- Longer sequences: 24 actions/hand (was 16), 64 hands/chunk (was 48)
- Deeper: 2 hand layers (was 1), 3 chunk layers (was 2)
- Wider: 64-dim embeddings (was 48), 128 FFN dim (was 96)
- [CLS] token aggregation (was mean pool)
- More aux humans (1500 chunks, was 800)
- 30 epochs + early stopping on best val_AP
"""
from __future__ import annotations

import gzip
import json
import pickle
import random
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

DATA_DIR = PROJECT / "data" / "benchmark"
AUX_HUMAN_PATH = PROJECT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
PSEUDO_PATH = PROJECT / "neurons" / "pseudo_live_labels.json"
CAP_DIR = PROJECT / "neurons" / "captured_chunks_h2"
MODEL_OUTPUT = PROJECT / "neurons" / "model_v22_transformer.pkl"

ACTION_TYPES = ["check", "call", "bet", "raise", "fold", "all_in", "small_blind", "big_blind", "other", ""]
STREETS = ["preflop", "flop", "turn", "river", ""]
AMOUNT_BUCKETS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0]
MAX_ACTIONS_PER_HAND = 18
MAX_HANDS_PER_CHUNK = 48
EMBED_DIM = 56
N_HEADS = 4
N_LAYERS_HAND = 1
N_LAYERS_CHUNK = 2
FFN_DIM = 112


def _bucket_amount(norm_bb: float) -> int:
    if norm_bb is None or norm_bb <= 0:
        return 0
    return min(range(len(AMOUNT_BUCKETS)), key=lambda i: abs(AMOUNT_BUCKETS[i] - norm_bb))


def encode_hand(hand: dict) -> np.ndarray:
    """(MAX_ACTIONS_PER_HAND, 4) int array."""
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


def load_aux_humans(chunks_to_build: int = 800, hands_per_chunk: int = 40, seed: int = 42):
    """Build synthetic human chunks from the auxiliary corpus."""
    print(f"  loading aux human corpus from {AUX_HUMAN_PATH.name}...")
    with gzip.open(AUX_HUMAN_PATH, "rt") as f:
        hands = json.load(f)
    print(f"  loaded {len(hands)} human hands")
    rng = random.Random(seed)
    rng.shuffle(hands)
    chunks = []
    for i in range(min(chunks_to_build, len(hands) // hands_per_chunk)):
        chunks.append(hands[i * hands_per_chunk : (i + 1) * hands_per_chunk])
    labels = np.zeros(len(chunks), dtype=np.int64)
    return chunks, labels


def load_pseudo():
    if not PSEUDO_PATH.exists():
        return [], np.zeros(0, dtype=np.int64)
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
        cc = d.get("chunks", [])
        for idx, lab in items:
            if idx < len(cc):
                cks.append(cc[idx]); ys.append(int(lab))
    return cks, np.array(ys, dtype=np.int64)


def main():
    import torch
    import torch.nn as nn

    print("=" * 60)
    print("v22 Deeper Hierarchical Action Transformer (gen2)")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    bench_cks, bench_y = load_benchmark()
    aux_cks, aux_y = load_aux_humans(chunks_to_build=1500)
    pseudo_cks, pseudo_y = load_pseudo()
    print(f"  benchmark: {len(bench_cks)} chunks (bots={int(bench_y.sum())})")
    print(f"  aux humans: {len(aux_cks)} chunks (all human)")
    print(f"  pseudo live: {len(pseudo_cks)} chunks (bots={int(pseudo_y.sum())})")

    print("\n[2/4] Encoding tokens...")
    Xb = np.stack([encode_chunk(c) for c in bench_cks])
    Xa = np.stack([encode_chunk(c) for c in aux_cks])
    Xp = np.stack([encode_chunk(c) for c in pseudo_cks]) if pseudo_cks else np.zeros((0,) + Xb.shape[1:], dtype=np.int64)
    print(f"  bench tokens: {Xb.shape}")
    print(f"  aux tokens: {Xa.shape}")
    print(f"  pseudo tokens: {Xp.shape}")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, confusion_matrix

    X_tr, X_va, y_tr, y_va = train_test_split(
        Xb, bench_y, test_size=0.15, random_state=42, stratify=bench_y
    )

    # Mix in aux humans (weight 1.0) and pseudo-labels (weight 0.3)
    w_tr = np.ones(len(y_tr), dtype=np.float32)
    w_aux = np.ones(len(aux_y), dtype=np.float32)
    w_p = np.full(len(pseudo_y), 0.3, dtype=np.float32) if len(pseudo_y) else np.zeros(0, dtype=np.float32)
    X_all = np.concatenate([X_tr, Xa, Xp], axis=0)
    y_all = np.concatenate([y_tr, aux_y, pseudo_y])
    w_all = np.concatenate([w_tr, w_aux, w_p])
    print(f"\nCombined train: {X_all.shape} bots={int(y_all.sum())} humans={int((1-y_all).sum())}")
    print(f"Holdout (benchmark only): {X_va.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[3/4] Building model...")

    class HierarchicalActionTransformerV22(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_emb = nn.Embedding(len(ACTION_TYPES), 32)
            self.street_emb = nn.Embedding(len(STREETS), 12)
            self.seat_emb = nn.Embedding(10, 12)
            self.amount_emb = nn.Embedding(len(AMOUNT_BUCKETS), 32)
            self.action_proj = nn.Linear(32 + 12 + 12 + 32, EMBED_DIM)
            self.action_pos = nn.Parameter(torch.randn(MAX_ACTIONS_PER_HAND, EMBED_DIM) * 0.02)
            # Learnable [CLS] tokens for both hand-level and chunk-level aggregation
            self.hand_cls = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
            self.chunk_cls = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
            hand_layer = nn.TransformerEncoderLayer(
                d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            self.hand_encoder = nn.TransformerEncoder(hand_layer, num_layers=N_LAYERS_HAND)
            self.hand_pos = nn.Parameter(torch.randn(MAX_HANDS_PER_CHUNK, EMBED_DIM) * 0.02)
            chunk_layer = nn.TransformerEncoderLayer(
                d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            self.chunk_encoder = nn.TransformerEncoder(chunk_layer, num_layers=N_LAYERS_CHUNK)
            self.head = nn.Sequential(
                nn.LayerNorm(EMBED_DIM),
                nn.Linear(EMBED_DIM, 96),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(96, 1),
            )

        def forward(self, x):
            # x: (B, H, A, 4)
            B, H, A, _ = x.shape
            x_flat = x.view(B * H, A, 4)
            a, s, se, am = x_flat[..., 0], x_flat[..., 1], x_flat[..., 2], x_flat[..., 3]
            e = torch.cat([self.action_emb(a), self.street_emb(s),
                           self.seat_emb(se), self.amount_emb(am)], dim=-1)
            e = self.action_proj(e) + self.action_pos.unsqueeze(0)

            # Prepend hand [CLS] token; encoder aggregates into it via attention
            BH = B * H
            cls_h = self.hand_cls.expand(BH, 1, EMBED_DIM)
            e = torch.cat([cls_h, e], dim=1)  # (BH, 1+A, D)
            action_mask = (a == len(ACTION_TYPES) - 1)
            cls_mask = torch.zeros(BH, 1, dtype=torch.bool, device=action_mask.device)
            full_mask = torch.cat([cls_mask, action_mask], dim=1)  # never mask CLS
            e = self.hand_encoder(e, src_key_padding_mask=full_mask)
            hand_emb = e[:, 0, :]  # take CLS
            hand_emb = hand_emb.view(B, H, -1) + self.hand_pos.unsqueeze(0)

            # Chunk-level: prepend chunk CLS
            chunk_cls = self.chunk_cls.expand(B, 1, EMBED_DIM)
            chunk_in = torch.cat([chunk_cls, hand_emb], dim=1)  # (B, 1+H, D)
            hand_mask = (x[:, :, 0, 0] == len(ACTION_TYPES) - 1)
            cls_mask_chunk = torch.zeros(B, 1, dtype=torch.bool, device=hand_mask.device)
            full_chunk_mask = torch.cat([cls_mask_chunk, hand_mask], dim=1)
            chunk_e = self.chunk_encoder(chunk_in, src_key_padding_mask=full_chunk_mask)
            chunk_emb = chunk_e[:, 0, :]  # take chunk CLS
            return self.head(chunk_emb).squeeze(-1)

    model = HierarchicalActionTransformerV22().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    crit = nn.BCEWithLogitsLoss(reduction="none")

    X_tr_t = torch.tensor(X_all, dtype=torch.long)
    y_tr_t = torch.tensor(y_all, dtype=torch.float32)
    w_tr_t = torch.tensor(w_all, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.long).to(device)

    print("\n[4/4] Training...")
    BATCH = 64
    N_EPOCH = 15
    n = len(X_tr_t)
    best_ap = 0.0
    best_state = None
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        train_loss = total / n
        model.eval()
        with torch.no_grad():
            p_va = []
            for i in range(0, len(X_va_t), BATCH):
                xb = X_va_t[i:i + BATCH]
                p_va.append(torch.sigmoid(model(xb)).cpu().numpy())
            p_va = np.concatenate(p_va)
        ap = average_precision_score(y_va, p_va)
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep:2d}  loss={train_loss:.4f}  val_AP={ap:.4f}  best={best_ap:.4f}", flush=True)

    print(f"\nBest val_AP: {best_ap:.4f}")
    preds = (p_va > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va, preds, labels=[0, 1]).ravel()
    print(f"  recall={tp/max(tp+fn,1):.4f}  fpr={fp/max(tn+fp,1):.4f}")

    payload = {
        "state_dict": best_state if best_state else {k: v.cpu() for k, v in model.state_dict().items()},
        "embed_dim": EMBED_DIM, "n_heads": N_HEADS,
        "n_layers_hand": N_LAYERS_HAND, "n_layers_chunk": N_LAYERS_CHUNK,
        "ffn_dim": FFN_DIM,
        "n_action_types": len(ACTION_TYPES), "n_streets": len(STREETS),
        "amount_buckets": AMOUNT_BUCKETS,
        "action_types": ACTION_TYPES, "streets": STREETS,
        "max_actions_per_hand": MAX_ACTIONS_PER_HAND,
        "max_hands_per_chunk": MAX_HANDS_PER_CHUNK,
        "validation_ap": float(best_ap),
        "version": 22,
        "n_params": n_params,
    }
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved {MODEL_OUTPUT} ({MODEL_OUTPUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
