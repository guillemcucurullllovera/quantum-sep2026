"""
C1.3 — One-Class SVM anomaly detector.

The OCSVM's role in the meta-agent is to be a *trust score* on the regime
HMM's output. When the OCSVM flags a month as anomalous, the downstream
Bayesian gate (C1.4) treats the regime classification with skepticism and
biases toward defensive positioning.

This is a deliberately complementary signal to the regime HMM:

    C1.1 HMM:   "What macro regime are we in?" (3-way classification)
    C1.3 OCSVM: "Is this month statistically improbable vs history?"
                (1-way novelty detection)

A month can be in a *known* regime (HMM confident) but *anomalous* (OCSVM
fires) — e.g. an unusual configuration of macro factors that wasn't in the
training distribution. That's exactly the situation where the regime HMM's
forecast should be down-weighted.

Implementation is a thin wrapper over sklearn.svm.OneClassSVM that adds:

1. Train-only z-score standardization baked into fit/predict, so callers
   don't have to remember to scale inputs. Crucial for RBF: without
   standardization the VIX feature (std ~8) dominates the kernel.
2. Continuous `anomaly_score()` = -decision_function so higher numbers
   mean more anomalous. The raw sklearn output has the opposite
   convention which is confusing for downstream consumers.
3. Bool `predict_anomalous()` returning True/False per row.
4. `save`/`load` via joblib with metadata.
5. `summary()` method exposing nu, gamma, support-vector count, and the
   anomaly threshold for downstream consumption.

The Phase 3 quantum challenger (qrbf-OCSVM, per P0.1 lineage) is a
drop-in replacement here: same fit/predict API, just swap the kernel
function. Keeping the wrapper sklearn-native for the baseline keeps that
substitution clean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.svm import OneClassSVM
import joblib


# ── data container ────────────────────────────────────────────────

@dataclass
class OCSVMSummary:
    """Human-readable summary of a fitted OCSVMAnomalyDetector."""
    n_train_samples: int
    n_features: int
    nu: float                  # target anomaly fraction
    gamma: str | float         # RBF bandwidth
    kernel: str
    n_support_vectors: int
    train_anomaly_rate: float  # observed fraction flagged anomalous on train
    feature_names: list
    feature_means: list        # train-only z-score parameters (for reproducibility)
    feature_stds: list

    def to_dict(self) -> dict:
        return asdict(self)


# ── main class ────────────────────────────────────────────────────

class OCSVMAnomalyDetector:
    """
    One-Class SVM with built-in train-only z-score standardization.

    Standard usage:

        det = OCSVMAnomalyDetector(nu=0.05, kernel="rbf", gamma="scale")
        det.fit(X_train)

        is_anom = det.predict_anomalous(X_test)   # bool array
        scores  = det.anomaly_score(X_test)       # float array, higher = more anomalous
    """

    def __init__(
        self,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        random_state: int = 42,
        feature_names: Optional[Sequence[str]] = None,
    ):
        if not (0.0 < nu < 1.0):
            raise ValueError(f"nu must be in (0, 1); got {nu}")
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.feature_names = list(feature_names) if feature_names else None

        self._model: Optional[OneClassSVM] = None
        self._mu: Optional[np.ndarray] = None      # train-only feature means
        self._sigma: Optional[np.ndarray] = None   # train-only feature stds
        self._fitted = False
        self._n_train_samples: int = 0
        self._n_features: int = 0

    # ── core API ──

    def fit(self, X: np.ndarray) -> "OCSVMAnomalyDetector":
        """Standardize features using train-only stats, then fit OneClassSVM."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, n_features); got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf; clean upstream before fit")

        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma = np.where(self._sigma > 1e-12, self._sigma, 1.0)
        Xz = (X - self._mu) / self._sigma

        # OneClassSVM has no random_state arg in older sklearn; ignore if unsupported
        self._model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self._model.fit(Xz)

        self._n_train_samples = X.shape[0]
        self._n_features = X.shape[1]
        self._fitted = True
        return self

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Apply the fit-time standardization parameters to new data."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"expected {self._n_features} features; got {X.shape[1]}"
            )
        return (X - self._mu) / self._sigma

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Continuous anomaly score: higher = more anomalous.

        Defined as `-decision_function`, since sklearn's OneClassSVM has
        positive `decision_function` for inliers and negative for outliers.
        We flip the sign so downstream code can use intuitive thresholds
        like `score > threshold`.
        """
        Xz = self._standardize(X)
        return -self._model.decision_function(Xz)

    def predict_anomalous(self, X: np.ndarray) -> np.ndarray:
        """Boolean array: True for anomalous rows, False for normal rows."""
        Xz = self._standardize(X)
        # sklearn returns +1 for inliers, -1 for outliers; we want True for outliers
        return self._model.predict(Xz) == -1

    # ── inspection ──

    def summary(self) -> OCSVMSummary:
        """Produce a JSON-serializable summary for the calibration report."""
        self._check_fitted()
        # Train-set anomaly rate (sanity check on nu)
        train_is_anom = self._model.predict(
            (self._train_X_cached() if hasattr(self, "_train_X_cached") else None)
        ) if False else None  # We didn't cache X; recompute support_vector ratio instead

        # Better: train_anomaly_rate = (# support vectors with alpha = nu * (1/(n*nu))) / n
        # Simpler and exact: re-evaluate on the support vectors' implied predictions.
        # We don't have X cached; compute the rate from the model directly:
        # The sklearn convention is that approximately nu * n samples are outliers.
        n_outliers_approx = self._model.fit_status_  # 0 if converged
        # Pull the training anomaly rate from the support vectors: |SV| / n is an
        # upper bound; exact rate needs the original X. We didn't cache it to
        # keep the model artifact small. Report the SV count as a proxy.

        return OCSVMSummary(
            n_train_samples=self._n_train_samples,
            n_features=self._n_features,
            nu=self.nu,
            gamma=self.gamma if not callable(self.gamma) else "callable",
            kernel=self.kernel,
            n_support_vectors=int(self._model.support_vectors_.shape[0]),
            train_anomaly_rate=float("nan"),  # computed by caller with .predict_anomalous(X_train)
            feature_names=self.feature_names or [],
            feature_means=self._mu.tolist(),
            feature_stds=self._sigma.tolist(),
        )

    # ── persistence ──

    def save(self, path: str | Path) -> None:
        """Pickle the fitted model + standardization params. Companion JSON summary."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "mu": self._mu,
            "sigma": self._sigma,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "n_train_samples": self._n_train_samples,
            "n_features": self._n_features,
        }
        joblib.dump(payload, path)
        summary_path = path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(self.summary().to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "OCSVMAnomalyDetector":
        """Load a previously-saved OCSVMAnomalyDetector."""
        payload = joblib.load(Path(path))
        inst = cls(
            nu=payload["nu"],
            kernel=payload["kernel"],
            gamma=payload["gamma"],
            random_state=payload["random_state"],
            feature_names=payload["feature_names"],
        )
        inst._model = payload["model"]
        inst._mu = payload["mu"]
        inst._sigma = payload["sigma"]
        inst._n_train_samples = payload["n_train_samples"]
        inst._n_features = payload["n_features"]
        inst._fitted = True
        return inst

    # ── internals ──

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "OCSVMAnomalyDetector has not been fitted; call .fit(X) first"
            )


__all__ = ["OCSVMAnomalyDetector", "OCSVMSummary"]
