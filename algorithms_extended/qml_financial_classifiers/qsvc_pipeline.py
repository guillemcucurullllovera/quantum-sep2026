"""QuantumSVC: a QSVC-style precomputed-quantum-kernel SVM wrapper.

Architectural pattern adapted from qiskit-machine-learning's
`qiskit_machine_learning.algorithms.classifiers.QSVC` (Apache-2.0): expose
an sklearn-like `fit(X, y)` / `predict(X)` / `score(X, y)` interface over a
quantum kernel, so the kernel choice is a plug-in rather than baked into the
training loop. No qiskit-machine-learning source is copied -- upstream QSVC
subclasses `sklearn.svm.SVC`; this platform has no sklearn dependency in its
venv, so the SVM solver here is `algorithms.pqk.kernels_gen1.SimpleSVM`
(Asymmetrica's own from-scratch SMO implementation)
extended with weighted-margin support (see `WeightedSimpleSVM` below) --
upstream's `class_weight=` convenience has no equivalent in SimpleSVM, so
that extension is original work, not adapted from upstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from algorithms.pqk.kernels import fidelity_gram, projected_gram
from algorithms.pqk.kernels_gen1 import SimpleSVM
from core.errors import ValidationError


class WeightedSimpleSVM(SimpleSVM):
    """SimpleSVM with per-sample box-constraint weighting: alpha_i is bounded
    by `C * sample_weight[i]` instead of a single global C, implementing the
    standard cost-sensitive soft-margin SVM formulation (Osuna et al. 1997)
    for class-imbalanced financial regime/signal data. This is a published,
    non-copyrightable technique -- not adapted from any specific library.
    """

    def fit(self, K: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> int:  # type: ignore[override]
        if sample_weight is None:
            return super().fit(K, y)
        n = len(y)
        if len(sample_weight) != n:
            raise ValidationError("sample_weight must match len(y)")
        self._per_sample_c = self.C * np.asarray(sample_weight, dtype=float)
        try:
            return self._fit_weighted(K, y)
        finally:
            del self._per_sample_c

    def _fit_weighted(self, K: np.ndarray, y: np.ndarray) -> int:
        # Same SMO structure as SimpleSVM.fit (kernels_gen1.py), with the
        # single box constraint `self.C` replaced by a per-sample bound
        # `c[i] = self.C * sample_weight[i]` everywhere it's used as an
        # upper bound on alpha_i or alpha_j -- the textbook cost-sensitive
        # soft-margin SVM extension (Osuna et al. 1997).
        n = len(y)
        y = y.astype(float)
        self.alpha = np.zeros(n)
        self.b = 0.0
        self.y_train = y.copy()
        c = self._per_sample_c
        rng = np.random.default_rng(0)

        for _ in range(self.max_iter):
            num_changed = 0
            for i in range(n):
                e_i = float(np.dot(self.alpha * y, K[i]) + self.b - y[i])
                if (y[i] * e_i < -self.tol and self.alpha[i] < c[i]) or (
                    y[i] * e_i > self.tol and self.alpha[i] > 0
                ):
                    j = i
                    while j == i:
                        j = int(rng.integers(n))
                    e_j = float(np.dot(self.alpha * y, K[j]) + self.b - y[j])
                    ai_old, aj_old = self.alpha[i], self.alpha[j]
                    if y[i] != y[j]:
                        lo = max(0.0, aj_old - ai_old)
                        hi = min(c[j], c[i] + aj_old - ai_old)
                    else:
                        lo = max(0.0, ai_old + aj_old - c[i])
                        hi = min(c[j], ai_old + aj_old)
                    if hi - lo < 1e-10:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    aj_new = np.clip(aj_old - y[j] * (e_i - e_j) / eta, lo, hi)
                    if abs(aj_new - aj_old) < 1e-8:
                        continue
                    ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)
                    b1 = self.b - e_i - y[i] * (ai_new - ai_old) * K[i, i] - y[j] * (aj_new - aj_old) * K[i, j]
                    b2 = self.b - e_j - y[i] * (ai_new - ai_old) * K[i, j] - y[j] * (aj_new - aj_old) * K[j, j]
                    if 0 < ai_new < c[i]:
                        self.b = b1
                    elif 0 < aj_new < c[j]:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    self.alpha[i], self.alpha[j] = ai_new, aj_new
                    num_changed += 1
            if num_changed == 0:
                break
        return int(np.sum(self.alpha > 1e-6))


@dataclass
class QuantumSVC:
    """sklearn-like precomputed-quantum-kernel classifier.

    Binary labels only (+-1); for multiclass, compose with a one-vs-rest
    loop at the caller level (matches this platform's existing VQC
    convention in algorithms.vqc_qml.training.RidgeReadout).
    """

    feature_map: str = "AngleCNOTRingV2"
    kernel_kind: str = "fidelity"  # "fidelity" | "projected"
    C: float = 1.0
    class_weight: str | None = None  # None | "balanced"
    _svm: WeightedSimpleSVM = field(init=False, repr=False)
    _X_train: np.ndarray = field(init=False, repr=False)

    def _gram(self, X: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        if self.kernel_kind == "fidelity":
            return fidelity_gram(X, self.feature_map, X2)
        if self.kernel_kind == "projected":
            return projected_gram(X, self.feature_map, X2)
        raise ValidationError(f"Unknown kernel_kind={self.kernel_kind!r}")

    def _sample_weights(self, y: np.ndarray) -> np.ndarray | None:
        if self.class_weight != "balanced":
            return None
        classes, counts = np.unique(y, return_counts=True)
        freq = dict(zip(classes, counts))
        return np.array([len(y) / (len(classes) * freq[label]) for label in y])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumSVC":
        if set(np.unique(y)) - {-1, 1}:
            raise ValidationError("QuantumSVC.fit expects binary labels in {-1, +1}")
        self._X_train = X
        K = self._gram(X)
        self._svm = WeightedSimpleSVM(C=self.C)
        self._svm.fit(K, y, sample_weight=self._sample_weights(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # _gram(X, X2=X_train) already returns shape (n_test, n_train),
        # matching what predict_kernel's K_test @ alpha expects.
        K_test = self._gram(X, self._X_train)
        return self._svm.predict_kernel(K_test)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
