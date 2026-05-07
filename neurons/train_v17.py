"""Train v17 — first model with REAL ground truth labels (1440 chunks).

The Poker44 training benchmark API released ground-truth-labeled chunks
on 2026-05-06. This script trains a fresh ensemble using those labels.

Compared to v14 (pseudo-label trained):
- v14: 8000+ synthetic + 712 pseudo-labeled chunks
- v17: 1440 REAL labeled chunks (perfectly 50/50 balanced)

Quality > quantity. v14 trained against itself; v17 trains against the
actual evaluator's ground truth — should significantly outperform.

Architecture: XGBoost classifier + LightGBM classifier ensemble.
Calibrated via isotonic. Held-out 20% for validation.
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurons.v14_features import extract_v14_features, V14_NAMES

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "benchmark"
MODEL_OUTPUT = PROJECT / "neurons" / "model_v17.pkl"


def load_benchmark_data():
    """Load all benchmark releases into (chunks, labels)."""
    all_chunks = []
    all_labels = []
    for f in sorted(DATA_DIR.glob("benchmark_*.json")):
        data = json.loads(f.read_text())
        super_chunks = data["data"]["chunks"]
        for sc in super_chunks:
            sub_chunks = sc.get("chunks", [])
            gt = sc.get("groundTruth", [])
            assert len(sub_chunks) == len(gt), f"Mismatch in {f.name}"
            for chunk, label in zip(sub_chunks, gt):
                all_chunks.append(chunk)
                all_labels.append(int(label))
    return all_chunks, np.array(all_labels, dtype=np.int32)


def extract_features(chunks):
    """Extract v14 (355) features for each chunk."""
    X = []
    for c in chunks:
        f = extract_v14_features(c)
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        X.append(f)
    return np.array(X, dtype=np.float32)


def train_ensemble(X_train, y_train, X_val, y_val):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import average_precision_score, confusion_matrix
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    print(f"\nTrain: {X_train.shape}  bots={int(y_train.sum())}  humans={int((1-y_train).sum())}")
    print(f"Val:   {X_val.shape}  bots={int(y_val.sum())}  humans={int((1-y_val).sum())}")

    print("\n=== XGBoost (calibrated isotonic) ===")
    xgb = CalibratedClassifierCV(
        estimator=XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.3, reg_lambda=2.0, gamma=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
        method="isotonic", cv=5,
    )
    xgb.fit(X_train, y_train)
    p_xgb_val = xgb.predict_proba(X_val)[:, 1]
    ap_xgb = average_precision_score(y_val, p_xgb_val)
    preds = (p_xgb_val > 0.5).astype(int)
    cm = confusion_matrix(y_val, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"  XGBoost val AP={ap_xgb:.4f}  recall={tp/max(tp+fn,1):.4f}  fpr={fp/max(tn+fp,1):.4f}")

    print("\n=== LightGBM (calibrated isotonic) ===")
    lgbm = CalibratedClassifierCV(
        estimator=LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.7, min_child_samples=15,
            reg_alpha=0.3, reg_lambda=2.0, num_leaves=31,
            random_state=43, n_jobs=-1, verbose=-1,
        ),
        method="isotonic", cv=5,
    )
    lgbm.fit(X_train, y_train)
    p_lgbm_val = lgbm.predict_proba(X_val)[:, 1]
    ap_lgbm = average_precision_score(y_val, p_lgbm_val)
    preds = (p_lgbm_val > 0.5).astype(int)
    cm = confusion_matrix(y_val, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"  LightGBM val AP={ap_lgbm:.4f}  recall={tp/max(tp+fn,1):.4f}  fpr={fp/max(tn+fp,1):.4f}")

    # Search ensemble weight
    best_w, best_ap = 0.5, 0.0
    for w in np.arange(0.2, 0.85, 0.05):
        ens = w * p_xgb_val + (1 - w) * p_lgbm_val
        ap = average_precision_score(y_val, ens)
        if ap > best_ap:
            best_ap, best_w = ap, w

    p_ens_val = best_w * p_xgb_val + (1 - best_w) * p_lgbm_val
    preds = (p_ens_val > 0.5).astype(int)
    cm = confusion_matrix(y_val, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = tp / max(tp+fn, 1)
    fpr = fp / max(tn+fp, 1)
    pen = 0 if fpr >= 0.10 else max(0, 1-fpr)**2
    reward = (0.65 * best_ap + 0.35 * recall) * pen
    print(f"\n=== Ensemble (XGB w={best_w:.2f}, LGBM w={1-best_w:.2f}) ===")
    print(f"  val AP={best_ap:.4f}  recall={recall:.4f}  fpr={fpr:.4f}")
    print(f"  Predicted reward (formula): {reward:.4f}")

    return xgb, lgbm, best_w, best_ap


def main():
    print("=" * 60)
    print("Poker44 v17 Training (REAL ground truth labels)")
    print("=" * 60)

    chunks, labels = load_benchmark_data()
    print(f"Loaded {len(chunks)} labeled chunks (bots={int(labels.sum())}, humans={int((1-labels).sum())})")

    print("\nExtracting v14 features...")
    X = extract_features(chunks)
    print(f"Feature matrix: {X.shape}")

    # 80/20 split, stratified
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.20, random_state=42, stratify=labels
    )

    xgb, lgbm, w, val_ap = train_ensemble(X_train, y_train, X_val, y_val)

    # Save
    payload = {
        "xgb_model": xgb,
        "lgbm_model": lgbm,
        "ensemble_weight_xgb": float(w),
        "feature_names": V14_NAMES,
        "version": 17,
        "training_size": int(len(y_train)),
        "validation_ap": float(val_ap),
        "labels_method": "real-ground-truth",
    }
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved {MODEL_OUTPUT} ({MODEL_OUTPUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
