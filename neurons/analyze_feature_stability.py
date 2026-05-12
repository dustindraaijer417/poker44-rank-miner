"""Step 1: Identify features that drift between benchmark (train) and live data.

For each of the 355 v14 features, compute mean/std on:
- benchmark chunks (with labels)
- live captured chunks (no labels but representative distribution)

Features with massive train/live shift are noise on inference. Drop them.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/subnet126/Poker44-subnet")
from neurons.v14_features import extract_v14_features, V14_NAMES

REPO = Path("/root/subnet126/Poker44-subnet")
BENCH_DIR = REPO / "data" / "benchmark"
CAP_DIR = REPO / "neurons" / "captured_chunks_h2"


def load_benchmark_chunks():
    chunks = []
    for f in sorted(BENCH_DIR.glob("benchmark_*.json")):
        d = json.loads(f.read_text())
        for sc in d["data"]["chunks"]:
            for c in sc.get("chunks", []):
                chunks.append(c)
    return chunks


def load_live_chunks():
    chunks = []
    for f in sorted(CAP_DIR.glob("query_*.json")):
        d = json.loads(f.read_text())
        chunks.extend(d.get("chunks", []))
    return chunks


def extract(chunks):
    X = []
    for c in chunks:
        if not c:
            X.append(np.zeros(355, dtype=np.float32))
        else:
            f = extract_v14_features(c)
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            X.append(f)
    return np.array(X, dtype=np.float32)


def main():
    bench = load_benchmark_chunks()
    live = load_live_chunks()
    print(f"benchmark chunks: {len(bench)}")
    print(f"live chunks: {len(live)}")

    Xb = extract(bench)
    Xl = extract(live)

    # Per-feature mean and std on both
    mb, sb = Xb.mean(axis=0), Xb.std(axis=0) + 1e-9
    ml, sl = Xl.mean(axis=0), Xl.std(axis=0) + 1e-9
    # Standardized mean shift: |mean_live - mean_bench| / std_bench
    shift = np.abs(ml - mb) / sb

    # Bench feature has any signal? If std=0, useless.
    useful_bench = sb > 0.01
    print(f"\nFeatures with >0.01 std on benchmark: {useful_bench.sum()}/355")

    # Rank features by stability (low shift = stable)
    stable = []
    unstable = []
    for i in range(355):
        if not useful_bench[i]:
            continue
        if shift[i] < 0.5:
            stable.append((i, V14_NAMES[i], shift[i], mb[i], sb[i], ml[i], sl[i]))
        elif shift[i] > 2.0:
            unstable.append((i, V14_NAMES[i], shift[i], mb[i], sb[i], ml[i], sl[i]))

    print(f"\n=== STABLE features (shift < 0.5): {len(stable)} ===")
    for i, name, s, mb_, sb_, ml_, sl_ in sorted(stable, key=lambda x: x[2])[:15]:
        print(f"  [{i:3d}] shift={s:.2f}  {name[:40]:40s}  b(m={mb_:.3f},s={sb_:.3f})  l(m={ml_:.3f},s={sl_:.3f})")
    print(f"  ... (showing 15 of {len(stable)})")

    print(f"\n=== UNSTABLE features (shift > 2.0): {len(unstable)} ===")
    for i, name, s, mb_, sb_, ml_, sl_ in sorted(unstable, key=lambda x: -x[2])[:15]:
        print(f"  [{i:3d}] shift={s:.2f}  {name[:40]:40s}  b(m={mb_:.3f},s={sb_:.3f})  l(m={ml_:.3f},s={sl_:.3f})")
    print(f"  ... (showing 15 of {len(unstable)})")

    # Save the stable feature indices to disk
    stable_idx = sorted([s[0] for s in stable])
    Path(REPO / "neurons" / "stable_features.json").write_text(
        json.dumps({"stable_indices": stable_idx, "count": len(stable_idx),
                    "total_features": 355, "rationale": "shift<0.5 between benchmark and live"}, indent=2)
    )
    print(f"\nSaved {len(stable_idx)} stable feature indices to neurons/stable_features.json")


if __name__ == "__main__":
    main()
