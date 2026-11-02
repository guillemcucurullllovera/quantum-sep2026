import numpy as np

from applications.regime_detection.synthetic_regime_dataset import SyntheticRegimeDataset
from applications.regime_detection.hmm_baseline_comparison import (
    best_matching_accuracy,
    run_hmm_vs_vqc_comparison,
)


def test_best_matching_accuracy_perfect_under_permutation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    # cluster labels are a relabeling (permutation) of y_true: 0->2, 1->0, 2->1
    perm_map = {0: 2, 1: 0, 2: 1}
    cluster_labels = np.array([perm_map[v] for v in y_true])
    acc = best_matching_accuracy(y_true, cluster_labels, n_classes=3)
    assert acc == 1.0  # a pure permutation must score perfectly once matched


def test_best_matching_accuracy_below_one_for_real_errors():
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    cluster_labels = np.array([0, 0, 1, 1, 1, 1, 2, 2, 0])  # 2 misassigned
    acc = best_matching_accuracy(y_true, cluster_labels, n_classes=3)
    assert 0.0 < acc < 1.0


def test_hmm_vs_vqc_comparison_runs_and_reports_real_numbers():
    ds = SyntheticRegimeDataset(n_per_regime=25, seed=21)
    results = run_hmm_vs_vqc_comparison(ds, n_qubits=3, n_splits=2)
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r.vqc_accuracy <= 1.0
        assert 0.0 <= r.hmm_best_matched_accuracy <= 1.0
        assert 0.0 <= r.hmm_random_baseline_accuracy <= 1.0
        # the Hungarian-matched HMM accuracy must be >= its own random baseline's
        # *expected* value structurally isn't guaranteed every single fold, but
        # matched accuracy is never less than the single best-class-frequency floor
        assert r.hmm_best_matched_accuracy >= 1.0 / 3.0 - 1e-9
