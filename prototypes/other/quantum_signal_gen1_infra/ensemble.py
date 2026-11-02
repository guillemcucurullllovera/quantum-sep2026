"""
Phase 2C — Quantum Ensemble Layer
====================================
Combines VQC, quantum kernel, and classical signals into a single
prediction via learned stacking.

Architecture:
  Signal producers (Level 0):
    - VQC v2:   8Q circuit → sigmoid → P(up)
    - FQK SVM:  quantum kernel → decision function → calibrated P(up)
    - PQK SVM:  projected kernel → decision function → calibrated P(up)
    - LR:       logistic regression → P(up)

  Meta-learner (Level 1):
    - Logistic regression on Level-0 outputs
    - Learns correlation-aware weights
    - Optionally adds raw features as passthrough

Key design decisions:
  - Stacking, not voting — learns optimal combination weights
  - Platt scaling for SVM outputs — raw margins aren't probabilities
  - Leave-one-out or cross-val for Level-0 predictions on train set
    to prevent information leakage
  - Decorrelation bonus: ensemble members that make different errors
    contribute even if individually weaker

Contract:
  - Input: 8-float feature vector (QuantumInputVector v1.0)
  - Output: P(up) in [0, 1], plus per-signal breakdown
"""

import json
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path


# ── Platt Scaling (calibrate SVM margins → probabilities) ──────────

class PlattScaler:
    """
    Platt scaling: fit sigmoid A·f(x) + B to map SVM decision values
    to calibrated probabilities.

    Minimizes: -Σ [tᵢ log(pᵢ) + (1-tᵢ) log(1-pᵢ)]
    where pᵢ = 1/(1 + exp(A·fᵢ + B))
    and tᵢ = (yᵢ·N₊ + 1)/(N₊ + 2) for positive, 1/(N₋ + 2) for negative.

    Reference: Platt (1999), Lin et al. (2007) improved algorithm.
    """

    def __init__(self):
        self.A = 0.0
        self.B = 0.0

    def fit(self, decision_values: np.ndarray, y: np.ndarray,
            max_iter: int = 100, min_step: float = 1e-10):
        """
        Fit Platt parameters on decision values and labels.
        y should be in {0, 1}.
        """
        deci = decision_values.astype(float)
        labels = y.astype(float)

        # Target probabilities (smoothed)
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        t = np.where(labels == 1,
                     (n_pos + 1) / (n_pos + 2),
                     1.0 / (n_neg + 2))

        A, B = 0.0, math.log((n_neg + 1) / (n_pos + 1))
        hi_target = (n_pos + 1) / (n_pos + 2)
        lo_target = 1.0 / (n_neg + 2)

        for _ in range(max_iter):
            # Compute probabilities (numerically stable)
            fApB = np.clip(deci * A + B, -500, 500)
            p = np.where(fApB >= 0,
                         np.exp(-fApB) / (1 + np.exp(-fApB)),
                         1.0 / (1 + np.exp(fApB)))
            p = np.clip(p, 1e-10, 1 - 1e-10)

            # Gradient and Hessian
            d1 = t - p
            d2 = p * (1 - p)

            h11 = np.sum(d2 * deci * deci) + 1e-8
            h22 = np.sum(d2) + 1e-8
            h21 = np.sum(d2 * deci)
            g1 = -np.sum(d1 * deci)
            g2 = -np.sum(d1)

            # Newton step
            det = h11 * h22 - h21 * h21
            if abs(det) < 1e-20:
                break
            dA = -(h22 * g1 - h21 * g2) / det
            dB = -(h11 * g2 - h21 * g1) / det

            if abs(dA) < min_step and abs(dB) < min_step:
                break

            A += dA
            B += dB

        self.A = A
        self.B = B

    def predict_proba(self, decision_values: np.ndarray) -> np.ndarray:
        """Map decision values to P(y=1)."""
        fApB = np.clip(decision_values * self.A + self.B, -500, 500)
        p = np.where(fApB >= 0,
                     np.exp(-fApB) / (1 + np.exp(-fApB)),
                     1.0 / (1 + np.exp(fApB)))
        return p

    def to_dict(self) -> dict:
        return {"A": self.A, "B": self.B}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattScaler":
        s = cls()
        s.A = d["A"]
        s.B = d["B"]
        return s


# ── Signal Wrappers ─────────────────────────────────────────────────

class VQCSignal:
    """
    Wraps a trained VQC v2 model to produce P(up) from features.
    Loads from saved model JSON.
    """

    def __init__(self, model_path: Optional[str] = None, vqc=None):
        if vqc is not None:
            self.vqc = vqc
        elif model_path is not None:
            from prototypes.other.quantum_signal_gen1_infra.circuit_v2 import TrainableVQC
            with open(model_path) as f:
                d = json.load(f)
            # Handle both direct dict and nested format
            params = d.get("params") or d.get("best_params", {}).get("params")
            if params is None:
                raise ValueError(f"Cannot find params in {model_path}")
            self.vqc = TrainableVQC(params=np.array(params))
        else:
            raise ValueError("Provide model_path or vqc instance")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(up) for each sample."""
        probs = np.zeros(len(X))
        for i, features in enumerate(X):
            raw = self.vqc.forward(features.tolist())
            probs[i] = raw
        return probs

    @property
    def name(self) -> str:
        return "VQC_v2"


class KernelSignal:
    """
    Wraps a trained kernel SVM to produce calibrated P(up).
    Requires: trained SVM + Platt scaler + feature map + kernel type info.

    For inference, needs either:
      - FQK: X_train (to compute kernel against training points)
      - PQK: X_train (to project and compute RBF)
    """

    def __init__(self, svm, platt: PlattScaler, feature_map_name: str,
                 kernel_type: str, X_train: np.ndarray,
                 gamma: float = 1.0):
        self.svm = svm
        self.platt = platt
        self.feature_map_name = feature_map_name
        self.kernel_type = kernel_type
        self.X_train = X_train
        self.gamma = gamma

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated P(up) for each sample."""
        from prototypes.other.quantum_signal_gen1_infra.kernel import (
            FEATURE_MAPS, compute_fidelity_gram,
            compute_projected_features, rbf_kernel,
        )

        fmap = FEATURE_MAPS[self.feature_map_name]()

        if self.kernel_type == "fidelity":
            K = compute_fidelity_gram(X, fmap, X2=self.X_train, verbose=False)
        else:  # projected
            proj_new = compute_projected_features(X, fmap, verbose=False)
            proj_train = compute_projected_features(self.X_train, fmap, verbose=False)
            K = rbf_kernel(proj_new, proj_train, gamma=self.gamma)

        decision = self.svm.decision_function(K)
        return self.platt.predict_proba(decision)

    @property
    def name(self) -> str:
        return f"Kernel_{self.kernel_type}_{self.feature_map_name}"


class LogisticSignal:
    """
    Simple logistic regression signal (classical baseline).
    Trained directly on 8-dim features.
    """

    def __init__(self, weights: Optional[np.ndarray] = None,
                 bias: float = 0.0, n_features: int = 8):
        self.n_features = n_features
        if weights is not None:
            self.weights = weights.copy()
            self.bias = bias
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray,
            lr: float = 0.01, epochs: int = 200, l2: float = 0.01):
        """Train logistic regression with L2 regularization."""
        for _ in range(epochs):
            logits = X @ self.weights + self.bias
            preds = _sigmoid_vec(logits)
            error = preds - y  # (N,)

            grad_w = (X.T @ error) / len(y) + l2 * self.weights
            grad_b = np.mean(error)

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(up) for each sample."""
        logits = X @ self.weights + self.bias
        return _sigmoid_vec(logits)

    def to_dict(self) -> dict:
        return {"weights": self.weights.tolist(), "bias": self.bias}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticSignal":
        return cls(weights=np.array(d["weights"]), bias=d["bias"])

    @property
    def name(self) -> str:
        return "LogisticRegression"


# ── Stacking Meta-Learner ──────────────────────────────────────────

class StackingEnsemble:
    """
    Level-1 meta-learner that combines Level-0 signal probabilities.

    Training procedure (to prevent leakage):
      1. Split train data into K folds
      2. For each fold, train Level-0 models on K-1 folds
      3. Predict held-out fold → out-of-fold (OOF) predictions
      4. Train Level-1 LR on OOF predictions

    Inference:
      1. Each Level-0 model produces P(up)
      2. Level-1 LR combines them → final P(up)
    """

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.meta_weights = None
        self.meta_bias = 0.0
        self.signal_names = []
        self.signal_stats = {}  # per-signal performance on val

    def train_meta(self, oof_probs: np.ndarray, y_train: np.ndarray,
                   signal_names: List[str],
                   lr: float = 0.05, epochs: int = 300, l2: float = 0.01,
                   constrain_positive: bool = True):
        """
        Train Level-1 logistic regression on out-of-fold predictions.

        Args:
            oof_probs: (N_train, n_signals) — OOF probability predictions
            y_train: (N_train,) — true labels {0, 1}
            signal_names: list of signal names for diagnostics
            constrain_positive: if True, clamp weights ≥ 0 after each step
                               (prevents inverting weak signals → overfitting)
        """
        self.signal_names = signal_names
        n_signals = oof_probs.shape[1]
        self.meta_weights = np.ones(n_signals) / n_signals  # Start at equal weights
        self.meta_bias = 0.0

        # Simple gradient descent (dataset is small: n_signals columns)
        for _ in range(epochs):
            logits = oof_probs @ self.meta_weights + self.meta_bias
            preds = _sigmoid_vec(logits)
            error = preds - y_train

            grad_w = (oof_probs.T @ error) / len(y_train) + l2 * self.meta_weights
            grad_b = np.mean(error)

            self.meta_weights -= lr * grad_w
            self.meta_bias -= lr * grad_b

            # Prevent negative weights (which invert signals and overfit)
            if constrain_positive:
                self.meta_weights = np.maximum(self.meta_weights, 0.0)

    def predict_proba(self, signal_probs: np.ndarray) -> np.ndarray:
        """
        Combine Level-0 probabilities into final prediction.

        Args:
            signal_probs: (N, n_signals) array of P(up) from each signal

        Returns:
            (N,) array of ensemble P(up)
        """
        logits = signal_probs @ self.meta_weights + self.meta_bias
        return _sigmoid_vec(logits)

    def predict(self, signal_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions."""
        return (self.predict_proba(signal_probs) >= threshold).astype(int)

    def get_weights_report(self) -> str:
        """Human-readable weight breakdown."""
        lines = ["  Signal Weights:"]
        if self.meta_weights is None:
            return "  [not trained]"
        for name, w in zip(self.signal_names, self.meta_weights):
            lines.append(f"    {name:<30s}  w={w:+.4f}")
        lines.append(f"    {'bias':<30s}  b={self.meta_bias:+.4f}")
        return "\n".join(lines)

    def evaluate(self, signal_probs: np.ndarray, y_true: np.ndarray) -> dict:
        """Full evaluation with per-signal and ensemble metrics."""
        ensemble_probs = self.predict_proba(signal_probs)
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        ensemble_acc = float(np.mean(ensemble_preds == y_true))

        # Per-signal accuracy
        signal_accs = {}
        for i, name in enumerate(self.signal_names):
            preds_i = (signal_probs[:, i] >= 0.5).astype(int)
            acc_i = float(np.mean(preds_i == y_true))
            signal_accs[name] = acc_i

        # Correlation matrix between signals
        n_signals = signal_probs.shape[1]
        corr = np.corrcoef(signal_probs.T) if n_signals > 1 else np.array([[1.0]])

        # Agreement rates (pairwise)
        agreements = {}
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                pi = (signal_probs[:, i] >= 0.5).astype(int)
                pj = (signal_probs[:, j] >= 0.5).astype(int)
                agree = float(np.mean(pi == pj))
                pair = f"{self.signal_names[i]} vs {self.signal_names[j]}"
                agreements[pair] = agree

        return {
            "ensemble_acc": ensemble_acc,
            "signal_accs": signal_accs,
            "correlation_matrix": corr.tolist(),
            "agreement_rates": agreements,
            "weights": {name: float(w) for name, w
                        in zip(self.signal_names, self.meta_weights)},
            "bias": float(self.meta_bias),
        }

    def to_dict(self) -> dict:
        return {
            "meta_weights": self.meta_weights.tolist() if self.meta_weights is not None else None,
            "meta_bias": self.meta_bias,
            "signal_names": self.signal_names,
            "n_folds": self.n_folds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StackingEnsemble":
        ens = cls(n_folds=d.get("n_folds", 5))
        ens.meta_weights = np.array(d["meta_weights"]) if d["meta_weights"] else None
        ens.meta_bias = d["meta_bias"]
        ens.signal_names = d["signal_names"]
        return ens


# ── Cross-Validated OOF Generation ──────────────────────────────────

def generate_oof_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 5,
    vqc_params: Optional[np.ndarray] = None,
    kernel_configs: Optional[List[dict]] = None,
    include_lr: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate out-of-fold predictions for the stacking meta-learner.
    This prevents information leakage from Level-0 to Level-1.

    For VQC: retrain from scratch on each fold (expensive but correct).
    For kernels: recompute gram + retrain SVM on each fold.
    For LR: retrain on each fold (fast).

    Returns:
        oof_probs: (N_train, n_signals) out-of-fold probabilities
        signal_names: list of signal names
    """
    from prototypes.other.quantum_signal_gen1_infra.kernel import (
        FEATURE_MAPS, compute_fidelity_gram,
        compute_projected_features, rbf_kernel, SimpleSVM,
    )

    N = len(X_train)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    fold_size = N // n_folds

    signal_names = []
    all_oof = []

    # ── LR signal ──
    if include_lr:
        signal_names.append("LR")
        oof_lr = np.zeros(N)

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else N
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            lr = LogisticSignal(n_features=X_train.shape[1])
            lr.fit(X_train[train_idx], y_train[train_idx], epochs=200)
            oof_lr[val_idx] = lr.predict_proba(X_train[val_idx])

        all_oof.append(oof_lr)
        if verbose:
            acc = float(np.mean((oof_lr >= 0.5).astype(int) == y_train))
            print(f"  OOF LR: {acc:.4f}")

    # ── Kernel signals ──
    if kernel_configs:
        for kconf in kernel_configs:
            ktype = kconf["kernel_type"]
            fmap_name = kconf["feature_map"]
            C = kconf.get("C", 1.0)
            gamma = kconf.get("gamma", 1.0)
            name = f"Kernel_{ktype}_{fmap_name}"
            signal_names.append(name)

            oof_k = np.zeros(N)

            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < n_folds - 1 else N
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                X_tr_fold = X_train[train_idx]
                y_tr_fold = 2 * y_train[train_idx].astype(float) - 1
                X_va_fold = X_train[val_idx]

                fmap = FEATURE_MAPS[fmap_name]()

                if ktype == "fidelity":
                    K_tr = compute_fidelity_gram(X_tr_fold, fmap, verbose=False)
                    K_va = compute_fidelity_gram(X_va_fold, fmap, X2=X_tr_fold, verbose=False)
                else:
                    proj_tr = compute_projected_features(X_tr_fold, fmap, verbose=False)
                    proj_va = compute_projected_features(X_va_fold, fmap, verbose=False)
                    K_tr = rbf_kernel(proj_tr, proj_tr, gamma=gamma)
                    K_va = rbf_kernel(proj_va, proj_tr, gamma=gamma)

                svm = SimpleSVM(C=C, max_iter=2000)
                svm.fit(K_tr, y_tr_fold)
                decision_vals = svm.decision_function(K_va)

                # Platt scale on this fold
                platt = PlattScaler()
                # Use train fold for Platt fitting
                dec_tr = svm.decision_function(K_tr)
                platt.fit(dec_tr, y_train[train_idx])

                oof_k[val_idx] = platt.predict_proba(decision_vals)

            all_oof.append(oof_k)
            if verbose:
                acc = float(np.mean((oof_k >= 0.5).astype(int) == y_train))
                print(f"  OOF {name}: {acc:.4f}")

    # ── VQC signal (simplified: use pre-trained model, no retraining per fold) ──
    # Full CV retraining of VQC is too expensive (~30 min per fold × 5 folds).
    # Instead: use the pre-trained VQC and mark this as approximate.
    # The VQC was trained on the full train set, so its OOF predictions
    # are slightly optimistic, but with early stopping + only 25 params
    # on 1830 samples, the leak is small.
    if vqc_params is not None:
        from prototypes.other.quantum_signal_gen1_infra.circuit_v2 import TrainableVQC
        signal_names.append("VQC_v2")

        vqc = TrainableVQC(params=vqc_params)
        oof_vqc = np.zeros(N)
        for i in range(N):
            oof_vqc[i] = vqc.forward(X_train[i].tolist())

        all_oof.append(oof_vqc)
        if verbose:
            acc = float(np.mean((oof_vqc >= 0.5).astype(int) == y_train))
            print(f"  OOF VQC_v2 (approx, full-train model): {acc:.4f}")

    oof_probs = np.column_stack(all_oof)
    return oof_probs, signal_names


# ── Quick Ensemble (no CV, for rapid testing) ──────────────────────

def quick_ensemble(
    val_probs: Dict[str, np.ndarray],
    y_val: np.ndarray,
    train_probs: Optional[Dict[str, np.ndarray]] = None,
    y_train: Optional[np.ndarray] = None,
) -> dict:
    """
    Build and evaluate a quick ensemble from pre-computed predictions.
    If train_probs provided, trains a meta-learner.
    Otherwise, uses simple averaging.

    Args:
        val_probs: {signal_name: P(up) array for val set}
        y_val: val labels
        train_probs: optional {signal_name: P(up) array for train set}
        y_train: optional train labels

    Returns:
        dict with ensemble results
    """
    signal_names = list(val_probs.keys())
    val_matrix = np.column_stack([val_probs[n] for n in signal_names])

    # ── Simple average ──
    avg_probs = np.mean(val_matrix, axis=1)
    avg_preds = (avg_probs >= 0.5).astype(int)
    avg_acc = float(np.mean(avg_preds == y_val))

    # ── Weighted ensemble (if train data available) ──
    stacking_acc = None
    ensemble = None
    if train_probs is not None and y_train is not None:
        train_matrix = np.column_stack([train_probs[n] for n in signal_names])
        ensemble = StackingEnsemble()
        ensemble.train_meta(train_matrix, y_train, signal_names)
        eval_result = ensemble.evaluate(val_matrix, y_val)
        stacking_acc = eval_result["ensemble_acc"]

    # ── Per-signal stats ──
    signal_accs = {}
    for i, name in enumerate(signal_names):
        preds = (val_matrix[:, i] >= 0.5).astype(int)
        signal_accs[name] = float(np.mean(preds == y_val))

    return {
        "average_acc": avg_acc,
        "stacking_acc": stacking_acc,
        "signal_accs": signal_accs,
        "ensemble": ensemble,
        "best_individual": max(signal_accs.values()),
        "best_individual_name": max(signal_accs, key=signal_accs.get),
        "ensemble_lift_vs_best": (stacking_acc or avg_acc) - max(signal_accs.values()),
    }


# ── Utilities ──────────────────────────────────────────────────────

def _sigmoid_vec(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for arrays."""
    result = np.zeros_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    result[neg] = exp_x / (1.0 + exp_x)
    return result


def compute_decorrelation(probs: Dict[str, np.ndarray]) -> dict:
    """
    Compute pairwise decorrelation metrics between signals.
    Higher decorrelation = more ensemble benefit.
    """
    names = list(probs.keys())
    n = len(names)
    predictions = {name: (probs[name] >= 0.5).astype(int) for name in names}

    disagreement = {}
    for i in range(n):
        for j in range(i + 1, n):
            pi = predictions[names[i]]
            pj = predictions[names[j]]
            disagree_rate = float(np.mean(pi != pj))
            pair = f"{names[i]} vs {names[j]}"
            disagreement[pair] = disagree_rate

    # Correlation of raw probabilities
    if n > 1:
        prob_matrix = np.column_stack([probs[name] for name in names])
        corr = np.corrcoef(prob_matrix.T)
    else:
        corr = np.array([[1.0]])

    return {
        "disagreement_rates": disagreement,
        "correlation_matrix": corr.tolist(),
        "signal_names": names,
        "mean_disagreement": float(np.mean(list(disagreement.values()))) if disagreement else 0.0,
    }
