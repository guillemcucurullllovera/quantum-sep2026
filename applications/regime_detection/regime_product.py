"""Regime detection product layer: a confidence-routed ensemble of
`classical.meta_agent.RegimeHMM` (unsupervised, principled posterior via
forward-backward smoothing) and any supervised classifier reachable through
`applications.classification.dispatcher` -- "when to trust which," made
concrete and measurable rather than left as a design question.

Why an ensemble, not just "pick the better one": `hmm_baseline_comparison
.py` already shows HMM and a supervised classifier score differently on the
same data (architecturally different: unsupervised clustering vs. labeled
classification). Neither dominates uniformly -- HMM's forward-backward
posterior gives a genuine confidence signal a hard classifier prediction
doesn't, but HMM's structure-only prior can misfire on regimes with
overlapping feature distributions. `run_regime_ensemble` routes each test
point to whichever model HMM itself claims most confidence in: when HMM's
posterior probability for its top state clears `confidence_threshold`, trust
HMM's (permutation-matched) call; otherwise fall back to the supervised
classifier. `hmm_only_accuracy` / `classifier_only_accuracy` report how each
source actually performed on the subset it was actually used for, so the
routing rule's value is measured, not assumed.

Leakage discipline: the cluster-to-regime-label permutation (HMM's states
are unordered from EM) is fit on a held-out *labeled slice of the training
set*, never on the test set -- unlike `hmm_baseline_comparison
.best_matching_accuracy`, which (as documented there) fits the permutation
against the test set itself as a standard clustering-evaluation protocol.
This module holds a stricter bar because it is meant as a deployable
product path, not a one-off diagnostic comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from applications.classification.dispatcher import dispatch
from applications.classification.errors import UnsupportedTaskError
from applications.classification.task_spec import Algorithm, ClassificationTaskSpec, TaskType
from classical.meta_agent.regime_hmm import RegimeHMM
from core.dataset_interfaces import FinancialDataset
from core.metrics import accuracy


def _fit_hmm_and_permutation(
    X_train: np.ndarray, y_train_idx: np.ndarray, seed: int, vix_col: int, match_fraction: float = 0.25
) -> tuple[RegimeHMM, dict[int, int]]:
    """Fit RegimeHMM on most of the training slice, then fit the
    cluster-id -> true-regime-index permutation on a held-out *labeled*
    slice of the training data (never touching the test set)."""
    n = len(y_train_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_match = max(3, int(n * match_fraction))
    match_idx, fit_idx = perm[:n_match], perm[n_match:]

    hmm = RegimeHMM(n_states=3, vix_col=vix_col, random_state=seed).fit(X_train[fit_idx])
    cluster_on_match = hmm.predict(X_train[match_idx])
    confusion = np.zeros((3, 3), dtype=int)
    for t, c in zip(y_train_idx[match_idx], cluster_on_match):
        confusion[t, c] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    cluster_to_true = {int(col): int(row) for row, col in zip(row_ind, col_ind)}
    return hmm, cluster_to_true


@dataclass
class RegimeEnsembleResult:
    ensemble_accuracy: float
    hmm_accuracy: float
    classifier_accuracy: float
    hmm_only_accuracy: float
    classifier_only_accuracy: float
    frac_routed_to_hmm: float
    confidence_threshold: float
    classifier_algorithm: str
    predictions: np.ndarray
    source: np.ndarray  # per-test-point: "hmm" or the classifier's algorithm value


def run_regime_ensemble(
    dataset: FinancialDataset,
    classifier_algorithm: Algorithm = Algorithm.VQC_V5_REUPLOAD,
    n_qubits: int = 6,
    seed: int = 0,
    confidence_threshold: float = 0.6,
    test_fraction: float = 0.3,
    vix_col: int = 0,
) -> RegimeEnsembleResult:
    """Confidence-routed HMM/classifier ensemble on a single chronological
    train/test split of `dataset` (exactly 3 regimes required)."""
    X, y = dataset.features(), dataset.targets()
    classes = np.unique(y)
    if len(classes) != 3:
        raise UnsupportedTaskError(f"regime ensemble requires exactly 3 regimes; got {len(classes)}")
    n_test = max(1, int(len(y) * test_fraction))
    n_train = len(y) - n_test
    if n_train < 6:
        raise UnsupportedTaskError("not enough training data to hold out an HMM match slice")
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = np.array([class_to_idx[v] for v in y_train])

    hmm, cluster_to_true = _fit_hmm_and_permutation(X_train, y_train_idx, seed, vix_col=vix_col)
    cluster_probs = hmm.predict_proba(X_test)
    cluster_labels = np.argmax(cluster_probs, axis=1)
    hmm_confidence = cluster_probs.max(axis=1)
    hmm_preds_idx = np.array([cluster_to_true.get(int(c), c) for c in cluster_labels])
    hmm_preds = classes[hmm_preds_idx]

    spec = ClassificationTaskSpec(task_type=TaskType.REGIME, algorithm=classifier_algorithm, n_qubits=n_qubits, seed=seed)
    clf_result = dispatch(spec, X_train, y_train, X_test, y_test)
    clf_preds = np.asarray(clf_result.predictions)

    trust_hmm = hmm_confidence >= confidence_threshold
    ensemble_preds = np.where(trust_hmm, hmm_preds, clf_preds)
    source = np.where(trust_hmm, "hmm", classifier_algorithm.value)

    hmm_only_acc = float(accuracy(y_test[trust_hmm], hmm_preds[trust_hmm])) if trust_hmm.any() else float("nan")
    clf_only_acc = (
        float(accuracy(y_test[~trust_hmm], clf_preds[~trust_hmm])) if (~trust_hmm).any() else float("nan")
    )

    return RegimeEnsembleResult(
        ensemble_accuracy=float(accuracy(y_test, ensemble_preds)),
        hmm_accuracy=float(accuracy(y_test, hmm_preds)),
        classifier_accuracy=clf_result.metric("accuracy"),
        hmm_only_accuracy=hmm_only_acc,
        classifier_only_accuracy=clf_only_acc,
        frac_routed_to_hmm=float(trust_hmm.mean()),
        confidence_threshold=confidence_threshold,
        classifier_algorithm=classifier_algorithm.value,
        predictions=ensemble_preds,
        source=source,
    )


__all__ = ["RegimeEnsembleResult", "run_regime_ensemble"]
