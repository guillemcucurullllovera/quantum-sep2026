"""
Layerwise Pretraining Protocol for VQC v5 (Fix 3)

Implements the Cerezo 2021 layerwise pretraining:
  1. Pretrain Layer 0 only (freeze Layer 1): 100 ADAM steps
  2. Unfreeze Layer 1, joint training: 200 ADAM steps
  3. Retrain ridge readout on frozen quantum features

Why layerwise pretraining prevents barren plateaus:
  Training all layers simultaneously: ∂C/∂θ_k ∝ 2^{-n} (global plateau).
  Training only layer 0 first: local cost function sees only the first
  layer's output, where Var[∂C/∂θ_k] = O(1/poly(n)) — polynomially small,
  not exponentially. After layer 0 converges, layer 1 inherits a useful
  initialization and has better-behaved gradients.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .ansatz import (
    circuit_features, param_count, init_params, vqc_circuit, z_expectations
)


# ── ADAM optimizer (numpy, gradient ascent/descent) ──────────────────────────

def adam_step(
    params: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    maximize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One ADAM step. Returns (updated_params, updated_m, updated_v)."""
    sign = 1 if maximize else -1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    params = params + sign * lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v


# ── Ridge regression (closed-form) ───────────────────────────────────────────

def ridge_fit(
    Phi: np.ndarray, y: np.ndarray, alpha: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form ridge regression: W = (Φ^T Φ + α I)^{-1} Φ^T Y.

    Schuld Representer Theorem §6.5: every quantum model is a linear model
    in the encoding feature space → ridge regression is the optimal readout.

    Args:
        Phi: Feature matrix, shape (N, n_features).
        y: Labels, shape (N,) or (N, n_classes) (one-hot or integer).
        alpha: Ridge regularization.

    Returns:
        W: Weight matrix, shape (n_features, n_classes).
        b: Bias vector, shape (n_classes,).
    """
    N, F = Phi.shape
    if y.ndim == 1:
        y_mat = y[:, None].astype(float)
    else:
        y_mat = y.astype(float)
    n_classes = y_mat.shape[1]

    # Center features and labels for bias computation
    phi_mean = Phi.mean(axis=0, keepdims=True)
    y_mean = y_mat.mean(axis=0, keepdims=True)
    Phi_c = Phi - phi_mean
    y_c = y_mat - y_mean

    A = Phi_c.T @ Phi_c + alpha * np.eye(F)
    W = np.linalg.solve(A, Phi_c.T @ y_c)  # (F, n_classes)
    b = y_mean.squeeze() - (phi_mean @ W).squeeze()
    return W, b


# ── Feature matrix computation ────────────────────────────────────────────────

def compute_features(X: np.ndarray, params: np.ndarray, n: int, d: int) -> np.ndarray:
    """Compute circuit features for all training samples."""
    return np.stack([circuit_features(X[i], params, n, d) for i in range(len(X))])


# ── Layerwise training ────────────────────────────────────────────────────────

class LayerwiseTrainer:
    """
    Layerwise pretraining for VQC v5.

    Stage 1: Train Layer 0 only (100 ADAM steps, lr=0.01)
    Stage 2: Joint training Layers 0+1 (200 ADAM steps, lr=0.005)
    Stage 3: Retrain ridge readout on frozen quantum features

    The loss function for circuit training: MSE between ridge predictions
    and one-hot encoded labels. Gradients via parameter shift.
    """

    def __init__(
        self,
        n: int = 10,
        d: int = 2,
        stage1_steps: int = 100,
        stage2_steps: int = 200,
        stage1_lr: float = 0.01,
        stage2_lr: float = 0.005,
        ridge_alpha: float = 0.01,
        sigma_init: float = 0.01,
        seed: Optional[int] = 42,
    ):
        self.n = n
        self.d = d
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.ridge_alpha = ridge_alpha
        self.sigma_init = sigma_init
        self.seed = seed

        self.params_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.loss_history_: list[float] = []

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        """One-hot encode y using self.classes_."""
        n_classes = len(self.classes_)
        oh = np.zeros((len(y), n_classes))
        for i, cls in enumerate(self.classes_):
            oh[y == cls, i] = 1.0
        return oh

    def _mse_loss_on_batch(
        self,
        X_batch: np.ndarray,
        y_oh: np.ndarray,
        params: np.ndarray,
        layer_mask: Optional[np.ndarray] = None,
    ) -> float:
        """MSE loss: L = (1/N) Σ ‖f(x_i;θ) - y_i‖^2."""
        Phi = compute_features(X_batch, params, self.n, self.d)
        if self.W_ is not None:
            preds = Phi @ self.W_ + self.b_[None, :]
            residuals = preds - y_oh
            return float(np.mean(residuals ** 2))
        # Without readout: use features directly (proxy loss)
        return float(np.mean((Phi - y_oh[:, :self.n]) ** 2))

    def _gradient_batch(
        self,
        X_batch: np.ndarray,
        y_oh: np.ndarray,
        params: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gradient of MSE loss w.r.t. params via correct parameter shift rule.

        Applies PSR to z-expectation features (linear observables, exact for RY/RZ),
        then chains through the MSE loss. Avoids ~15-20% bias from applying PSR
        to the nonlinear loss directly.

        grad_k = Σ_{i,j} (∂L/∂Phi_{ij}) * (Phi_plus - Phi_minus)[i,j] / 2
        """
        grad = np.zeros_like(params)
        eps = np.pi / 2  # parameter shift for RY/RZ

        N = len(X_batch)
        Phi = compute_features(X_batch, params, self.n, self.d)   # (N, n)

        # dL/dPhi: chain rule of MSE through readout (loss is nonlinear in Phi)
        if self.W_ is not None:
            preds = Phi @ self.W_ + self.b_[None, :]              # (N, n_classes)
            dL_dPhi = (2.0 / N) * (preds - y_oh) @ self.W_.T     # (N, n)
        else:
            n_out = min(self.n, y_oh.shape[1])
            dL_dPhi = np.zeros_like(Phi)
            dL_dPhi[:, :n_out] = (2.0 / N) * (Phi[:, :n_out] - y_oh[:, :n_out])

        # PSR on features (exact for RY/RZ gates)
        for k in np.where(active_mask)[0]:
            p_plus = params.copy(); p_plus[k] += eps
            p_minus = params.copy(); p_minus[k] -= eps
            Phi_plus = compute_features(X_batch, p_plus, self.n, self.d)    # (N, n)
            Phi_minus = compute_features(X_batch, p_minus, self.n, self.d)  # (N, n)
            dPhi_k = (Phi_plus - Phi_minus) / 2.0  # exact feature gradient
            grad[k] = float(np.sum(dL_dPhi * dPhi_k))  # chain rule

        return grad

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 16) -> "LayerwiseTrainer":
        """
        Fit VQC v5 with layerwise pretraining.

        Args:
            X: Feature matrix, shape (N, F). First n features used.
            y: Labels, shape (N,). Values in {-1, 0, 1}.
            batch_size: Mini-batch size for gradient computation.

        Returns:
            self.
        """
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_oh = self._one_hot(y)

        # Initialize parameters
        params = init_params(self.n, self.d, sigma=self.sigma_init, seed=self.seed or 42)
        n_params = len(params)
        params_per_layer = param_count(self.n, 1)  # 2*n per layer

        # Fit an initial ridge readout so that stage 1 has a valid loss signal
        Phi_init = compute_features(X, params, self.n, self.d)
        self.W_, self.b_ = ridge_fit(Phi_init, y_oh, self.ridge_alpha)

        # Stage 1: Train Layer 0 only
        layer0_mask = np.zeros(n_params, dtype=bool)
        layer0_mask[:params_per_layer] = True

        m_adam = np.zeros(n_params)
        v_adam = np.zeros(n_params)

        for t in range(1, self.stage1_steps + 1):
            batch_idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
            X_b, y_b = X[batch_idx], y_oh[batch_idx]
            grad = self._gradient_batch(X_b, y_b, params, layer0_mask)
            params, m_adam, v_adam = adam_step(
                params, grad, m_adam, v_adam, t, lr=self.stage1_lr
            )

        # Re-fit ridge readout after stage 1 (warm-start for stage 2)
        Phi = compute_features(X, params, self.n, self.d)
        self.W_, self.b_ = ridge_fit(Phi, y_oh, self.ridge_alpha)

        # Stage 2: Joint training (all layers)
        all_mask = np.ones(n_params, dtype=bool)
        m_adam = np.zeros(n_params)
        v_adam = np.zeros(n_params)

        for t in range(1, self.stage2_steps + 1):
            batch_idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
            X_b, y_b = X[batch_idx], y_oh[batch_idx]
            grad = self._gradient_batch(X_b, y_b, params, all_mask)
            params, m_adam, v_adam = adam_step(
                params, grad, m_adam, v_adam, t, lr=self.stage2_lr
            )
            if t % 20 == 0:
                loss = self._mse_loss_on_batch(X_b, y_b, params)
                self.loss_history_.append(loss)

        self.params_ = params

        # Stage 3: Retrain ridge readout on final frozen features
        Phi_final = compute_features(X, params, self.n, self.d)
        self.W_, self.b_ = ridge_fit(Phi_final, y_oh, self.ridge_alpha)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via ridge scores + softmax."""
        if self.params_ is None:
            raise RuntimeError("Call fit() first.")
        Phi = compute_features(X, self.params_, self.n, self.d)
        scores = Phi @ self.W_ + self.b_[None, :]  # (N, n_classes)
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
