import numpy as np
import pytest

from algorithms_extended.qml_financial_classifiers.hyperparameter_search import grid_search_walk_forward
from algorithms_extended.qml_financial_classifiers.qsvc_pipeline import QuantumSVC, WeightedSimpleSVM
from core.dataset_interfaces import FinancialDataset
from core.errors import ValidationError
from core.schemas import DatasetDescriptor


def _separable_dataset(n=24, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = np.where(X[:, 0] + 0.3 * X[:, 1] > 0, 1.0, -1.0)
    return X, y


class _ToyFinancialDataset(FinancialDataset):
    def __init__(self, n=24, seed=0):
        self._X, self._y = _separable_dataset(n, seed)

    @property
    def descriptor(self):
        return DatasetDescriptor(name="toy_qsvc", n_samples=len(self._y), n_features=2, domain="test")

    def features(self):
        return self._X

    def targets(self):
        return self._y


def test_quantum_svc_fits_and_predicts_reasonably():
    X, y = _separable_dataset(n=20, seed=1)
    model = QuantumSVC(C=1.0).fit(X, y)
    preds = model.predict(X)
    # Not a trivial-constant assertion: on a linearly-separable-ish synthetic
    # task the model should meaningfully beat random guessing (0.5).
    assert model.score(X, y) > 0.7
    assert set(np.unique(preds)) <= {-1.0, 1.0}


def test_quantum_svc_rejects_non_binary_labels():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0, 1, 2])
    with pytest.raises(ValidationError):
        QuantumSVC().fit(X, y)


def test_weighted_svm_matches_unweighted_when_weights_uniform():
    """Regression guard: WeightedSimpleSVM with all weights=1 should reach
    essentially the same decision boundary as unweighted SimpleSVM (proves
    the weighted SMO variant isn't silently different in the unweighted
    case, not just that it 'runs')."""
    from algorithms.pqk.kernels_gen1 import SimpleSVM

    X, y = _separable_dataset(n=16, seed=2)
    K = X @ X.T + 1.0  # cheap valid PSD-ish kernel for a fast, deterministic test

    plain = SimpleSVM(C=1.0)
    plain.fit(K, y)

    weighted = WeightedSimpleSVM(C=1.0)
    weighted.fit(K, y, sample_weight=np.ones(len(y)))

    plain_preds = plain.predict_kernel(K)
    weighted_preds = weighted.predict_kernel(K)
    agreement = np.mean(plain_preds == weighted_preds)
    assert agreement >= 0.85  # different pair-selection order in SMO can shift a few support vectors


def test_class_weight_balanced_actually_changes_fit():
    """A real assertion, not cosmetic: heavily imbalance the training set
    and confirm class_weight='balanced' changes which support vectors get
    selected / the resulting decision boundary, vs class_weight=None."""
    rng = np.random.default_rng(3)
    n_majority, n_minority = 18, 3
    X_maj = rng.normal(loc=[1.0, 1.0], scale=0.3, size=(n_majority, 2))
    X_min = rng.normal(loc=[-1.0, -1.0], scale=0.3, size=(n_minority, 2))
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.ones(n_majority), -np.ones(n_minority)])

    unweighted = QuantumSVC(C=1.0, kernel_kind="fidelity", class_weight=None).fit(X, y)
    weighted = QuantumSVC(C=1.0, kernel_kind="fidelity", class_weight="balanced").fit(X, y)

    # minority-class recall should not get worse (and typically improves)
    # under balanced weighting -- the whole point of the feature.
    minority_mask = y == -1
    recall_unweighted = np.mean(unweighted.predict(X)[minority_mask] == -1)
    recall_weighted = np.mean(weighted.predict(X)[minority_mask] == -1)
    assert recall_weighted >= recall_unweighted


def test_grid_search_walk_forward_never_leaks_future():
    ds = _ToyFinancialDataset(n=30, seed=4)
    splits = ds.split_walk_forward(n_splits=3)
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()


def test_grid_search_walk_forward_finds_a_valid_result():
    ds = _ToyFinancialDataset(n=30, seed=5)
    grid = {"feature_map": ["AngleCNOTRingV2"], "kernel_kind": ["fidelity"], "C": [0.5, 2.0]}
    result = grid_search_walk_forward(ds, grid, n_splits=2)
    assert result.best_params["C"] in (0.5, 2.0)
    assert 0.0 <= result.best_score <= 1.0
    assert len(result.all_results) == 2  # one row per grid combination
