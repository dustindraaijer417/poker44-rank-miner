"""Model classes for Poker44 miner (needed for pickling/unpickling)."""

import numpy as np


class RobustEnsemble:
    """Ensemble with quantile normalization and multi-model consensus."""

    def __init__(self, qt_models, raw_models, quantile_transformer, selected_indices, threshold=0.5):
        self.qt_models = qt_models
        self.raw_models = raw_models
        self.qt = quantile_transformer
        self.selected_indices = selected_indices
        self._optimal_threshold = threshold

    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        X_sel = X[:, self.selected_indices]
        X_qt = self.qt.transform(X_sel)

        probs = np.zeros((X.shape[0], 2))
        total_weight = 0

        for model in self.qt_models:
            w = 0.35
            probs += w * model.predict_proba(X_qt)
            total_weight += w

        for model in self.raw_models:
            w = 0.30
            probs += w * model.predict_proba(X)
            total_weight += w

        probs /= total_weight
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self._optimal_threshold).astype(int)


class CalibratedEnsemble:
    """Simple weighted ensemble of calibrated models."""

    def __init__(self, models, weights, threshold=0.5):
        self.models = models
        self.weights = weights
        self._optimal_threshold = threshold

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0] if hasattr(X, 'shape') else len(X), 2))
        for model, weight in zip(self.models, self.weights):
            probs += weight * model.predict_proba(X)
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self._optimal_threshold).astype(int)


class _EnsembleModel:
    """Legacy ensemble for v4/v5 models."""

    def __init__(self, model_a, model_b, weight_a=0.6):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self._optimal_threshold = 0.5

    def predict_proba(self, X):
        pa = self.model_a.predict_proba(X)
        pb = self.model_b.predict_proba(X)
        return self.weight_a * pa + (1 - self.weight_a) * pb


class LiveCalibratedEnsemble:
    """LightGBM + XGBoost ensemble trained on live + benchmark data (v7)."""

    def __init__(self, lgb_model, xgb_model):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self._optimal_threshold = 0.5

    def predict_proba(self, X):
        """Return (n_samples, 2) array of [P(human), P(bot)]."""
        lgb_p = self.lgb_model.predict_proba(X)[:, 1]
        xgb_p = self.xgb_model.predict_proba(X)[:, 1]
        bot_prob = (lgb_p + xgb_p) / 2.0
        return np.column_stack([1 - bot_prob, bot_prob])


class _TripleEnsemble:
    """Weighted average of three calibrated models (v11)."""

    def __init__(self, model_a, model_b, model_c, w_a=0.4, w_b=0.4, w_c=0.2, threshold=0.5):
        self.model_a = model_a  # XGBoost
        self.model_b = model_b  # LightGBM
        self.model_c = model_c  # RandomForest
        self.w_a = float(w_a)
        self.w_b = float(w_b)
        self.w_c = float(w_c)
        self._optimal_threshold = float(threshold)

    def predict_proba(self, X):
        pa = self.model_a.predict_proba(X)
        pb = self.model_b.predict_proba(X)
        pc = self.model_c.predict_proba(X)
        return self.w_a * pa + self.w_b * pb + self.w_c * pc

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._optimal_threshold).astype(int)


class _V14Ensemble:
    """v14 ensemble: separate ranker (LambdaMART) + classifier.

    The miner can choose which signal to use:
    - predict_proba(X)[:, 1] → calibrated bot probability (for binary prediction)
    - predict_rank(X)        → AP-optimized ranking score (normalized 0..1)
    """

    def __init__(self, classifier, ranker, ranker_min, ranker_max, threshold=0.5):
        self.classifier = classifier
        self.ranker = ranker
        self.ranker_min = float(ranker_min)
        self.ranker_max = float(ranker_max)
        self._optimal_threshold = float(threshold)

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values
        return self.classifier.predict_proba(X)

    def predict_rank(self, X):
        if hasattr(X, "values"):
            X = X.values
        raw = self.ranker.predict(X)
        denom = max(self.ranker_max - self.ranker_min, 1e-9)
        norm = (raw - self.ranker_min) / denom
        return np.clip(norm, 0.0, 1.0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._optimal_threshold).astype(int)


class _V12RobustEnsemble:
    """Hybrid ensemble for v12: 2 calibrated models on quantile-normalized features
    + 2 calibrated models on raw features. Quantile normalization handles V1
    distribution drift; raw features keep absolute-scale signal."""

    def __init__(self, qt_models, raw_models, qt_transformer, qt_weight=0.5, threshold=0.5):
        self.qt_models = qt_models
        self.raw_models = raw_models
        self.qt = qt_transformer
        self.qt_weight = float(qt_weight)
        self._optimal_threshold = float(threshold)

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values
        X_qt = self.qt.transform(X)
        n = X.shape[0]
        out = np.zeros((n, 2))
        if self.qt_models:
            qt_probs = np.zeros((n, 2))
            for m in self.qt_models:
                qt_probs += m.predict_proba(X_qt)
            qt_probs /= len(self.qt_models)
            out += self.qt_weight * qt_probs
        if self.raw_models:
            raw_probs = np.zeros((n, 2))
            for m in self.raw_models:
                raw_probs += m.predict_proba(X)
            raw_probs /= len(self.raw_models)
            out += (1 - self.qt_weight) * raw_probs
        return out

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._optimal_threshold).astype(int)
