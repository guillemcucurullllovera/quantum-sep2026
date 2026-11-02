import numpy as np

from applications.classification.synthetic_financial_dataset import SyntheticDirectionalDataset
from applications.classification.three_way_comparison import run_three_way_comparison

# Reduced-scale params for test speed (documented, not silently hidden): fewer
# samples/qubits and lighter optimizer budgets than the application defaults.
# The comparison logic under test does not depend on convergence quality.
_FAST = dict(n_qubits=3, n_splits=2, seed=0)


def test_three_way_comparison_runs_all_four_methods_and_reports_honestly():
    ds = SyntheticDirectionalDataset(n_samples=60, seed=3)
    results = run_three_way_comparison(ds, **_FAST)
    assert len(results) == 2
    for r in results:
        for acc in (r.vqc_accuracy, r.pqk_accuracy, r.qnn_accuracy, r.classical_accuracy):
            assert 0.0 <= acc <= 1.0
        assert r.best_method in ("vqc", "pqk", "qnn", "classical")


def test_best_method_is_actually_the_max():
    ds = SyntheticDirectionalDataset(n_samples=60, seed=3)
    results = run_three_way_comparison(ds, **_FAST)
    for r in results:
        accs = {
            "vqc": r.vqc_accuracy,
            "pqk": r.pqk_accuracy,
            "qnn": r.qnn_accuracy,
            "classical": r.classical_accuracy,
        }
        assert accs[r.best_method] == max(accs.values())
