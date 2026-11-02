"""Quantum (VQC) vs. classical (RegimeHMM) regime-detection comparison.

`classical.meta_agent.RegimeHMM` is an *unsupervised* 3-state Gaussian HMM
(fit via EM on features alone, no labels) with a fixed bull/bear/crisis
labeling convention keyed to a "VIX-like" column being lowest-mean = bull.
The platform's `SyntheticRegimeDataset` (trend_up/trend_down/ranging) is a
*labeled* dataset for a supervised VQC classifier, and its features are not
VIX-semantic -- so a direct accuracy comparison is not meaningful without
first resolving the label-permutation ambiguity inherent to any unsupervised
clustering method.

This module makes that comparison honest rather than skipping it: HMM's
Viterbi state assignment is compared to ground truth via the standard
technique for scoring unsupervised clustering against labels -- find the
best possible one-to-one state-to-label matching (Hungarian assignment on
the confusion matrix), then report accuracy under that best matching. This
is the correct way to ask "how good is HMM's regime structure," not a way to
flatter it -- the same matching is available to any clustering method, and
a matching that only slightly beats random permutation is reported as such.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.dataset_interfaces import FinancialDataset
from core.registry import APPLICATION_REGISTRY

from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from algorithms.vqc_qml.training import TrainingConfig, VQCTrainer
from classical.meta_agent.regime_hmm import RegimeHMM


def best_matching_accuracy(y_true: np.ndarray, cluster_labels: np.ndarray, n_classes: int) -> float:
    """Hungarian-matched accuracy: the best possible accuracy achievable by
    relabeling `cluster_labels` with a one-to-one permutation onto the true
    classes. Standard technique for scoring unsupervised clustering against
    ground truth (see e.g. scikit-learn's `contingency_matrix`-based approach).
    """
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, c in zip(y_true, cluster_labels):
        confusion[int(t), int(c)] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)  # maximize matched count
    matched = confusion[row_ind, col_ind].sum()
    return float(matched) / len(y_true)


@dataclass
class RegimeHMMComparisonFoldResult:
    fold: int
    vqc_accuracy: float
    hmm_best_matched_accuracy: float
    hmm_random_baseline_accuracy: float
    hmm_beats_random: bool
    hmm_beats_vqc: bool


def run_hmm_vs_vqc_comparison(
    dataset: FinancialDataset,
    n_qubits: int = 6,
    n_splits: int = 3,
) -> list[RegimeHMMComparisonFoldResult]:
    X, y = dataset.features(), dataset.targets()
    n_classes = len(np.unique(y))
    splits = dataset.split_walk_forward(n_splits=n_splits)
    ansatz = VQCAnsatz(n_qubits=n_qubits, version=VQCVersion.V5_REUPLOAD)

    results: list[RegimeHMMComparisonFoldResult] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        vqc = VQCTrainer(ansatz, TrainingConfig()).fit(X_train, y_train)
        vqc_acc = vqc.accuracy(X_test, y_test)

        # RegimeHMM is purpose-built for exactly 3 states; only run it when
        # the dataset's class count matches, otherwise report NaN honestly
        # rather than forcing a mismatched comparison.
        if n_classes != 3:
            results.append(
                RegimeHMMComparisonFoldResult(fold, vqc_acc, float("nan"), float("nan"), False, False)
            )
            continue

        hmm = RegimeHMM(n_states=3, vix_col=0, random_state=fold)
        hmm.fit(X_train)
        cluster_labels = hmm.predict(X_test)
        hmm_acc = best_matching_accuracy(y_test, cluster_labels, n_classes=3)

        # random baseline: expected accuracy of the best matching under a
        # uniform-random cluster assignment, estimated empirically (closed
        # form is available but this is simpler and equally honest)
        rng = np.random.default_rng(fold)
        random_scores = [
            best_matching_accuracy(y_test, rng.integers(0, 3, size=len(y_test)), n_classes=3)
            for _ in range(20)
        ]
        random_baseline = float(np.mean(random_scores))

        results.append(
            RegimeHMMComparisonFoldResult(
                fold=fold,
                vqc_accuracy=vqc_acc,
                hmm_best_matched_accuracy=hmm_acc,
                hmm_random_baseline_accuracy=random_baseline,
                hmm_beats_random=hmm_acc > random_baseline,
                hmm_beats_vqc=hmm_acc > vqc_acc,
            )
        )
    return results


@APPLICATION_REGISTRY.register("regime_detection_hmm_vs_vqc")
class RegimeHMMvsVQCApp:
    """Registered application entry point (see core.registry / core.cli)."""

    @staticmethod
    def run(**kwargs) -> list[RegimeHMMComparisonFoldResult]:
        from applications.regime_detection.synthetic_regime_dataset import SyntheticRegimeDataset

        dataset = kwargs.pop("dataset", None) or SyntheticRegimeDataset()
        return run_hmm_vs_vqc_comparison(dataset, **kwargs)
