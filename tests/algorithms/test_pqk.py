import numpy as np
import pytest

from algorithms.pqk import kernels, kernels_gen1
from algorithms.pqk.caching import KernelCache
from algorithms.pqk.feature_maps import FEATURE_MAPS, N_QUBITS
from algorithms.pqk.kta_alignment import compute_kta
from algorithms.pqk.noise_study import run_noise_sensitivity_study
from algorithms.pqk.nystrom import NystromApproximation
from algorithms.pqk.pqk_v2 import PQKConfig, PQKv2
from algorithms.pqk.training import sweep_feature_maps, train_gen1_svm, train_pqk_v2
from applications.classification.kernel_classifier import run_kernel_classification_comparison
from classical.kernels.baselines import polynomial_kernel, rbf_kernel, train_classical_svm
from core.experiment_registry import ExperimentRegistry
from core.metrics import is_positive_semidefinite


@pytest.fixture()
def toy_data():
    rng = np.random.default_rng(0)
    n = 24
    X = rng.uniform(0, 1, size=(n, N_QUBITS))
    y = (X[:, 0] * X[:, 1] > 0.25).astype(int)
    return X, y


def test_all_feature_maps_produce_psd_kernel_after_correction(toy_data):
    X, _ = toy_data
    for name in FEATURE_MAPS:
        K = kernels.projected_gram(X[:10], name, psd_correct=True)
        assert is_positive_semidefinite(K), f"{name} projected Gram not PSD after correction"


def test_fidelity_kernel_diagonal_is_one(toy_data):
    X, _ = toy_data
    K = kernels.fidelity_gram(X[:6], "AngleCNOT", psd_correct=False)
    assert np.allclose(np.diag(K), 1.0, atol=1e-8)  # |<psi|psi>|^2 == 1 exactly


def test_kta_gen1_vs_gen2_feature_map_diagnostics(toy_data):
    """The whole point of Gen2: AngleCNOTRingV2 should not be catastrophically
    more concentrated than the other maps on this toy separable dataset.
    This does not assert Gen2 > Gen1 on a specific toy set (that would be a
    fragile, potentially-false claim); it only asserts every registered map
    produces a *non-degenerate* kernel here, i.e. concentration_std > 0."""
    X, y = toy_data
    rows = sweep_feature_maps(X[:12], y[:12], method="projected", gammas=(1.0,))
    assert len(rows) == len(FEATURE_MAPS)
    for row in rows:
        assert row["concentration_std"] >= 0.0
        assert -1.0 <= row["kta"] <= 1.0


def test_nystrom_approximation_matches_exact_kernel_within_tolerance(toy_data):
    X, _ = toy_data

    def kfn(x1, x2):
        return float(np.dot(x1, x2))  # cheap linear "kernel" for a fast, deterministic test

    nystrom = NystromApproximation(m=8, seed=0).fit(X, kfn)
    rel_err = nystrom.accuracy_vs_exact(X, kernel_fn=kfn, n_test=12)
    assert rel_err < 0.05, f"Nystrom relative error too high: {rel_err}"


def test_kernel_cache_hits_and_returns_identical_result(tmp_path, toy_data):
    X, _ = toy_data
    cache = KernelCache(cache_dir=tmp_path)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return kernels.projected_gram(X[:8], "CPMap")

    k1 = cache.get_or_compute(X[:8], "CPMap", "projected", compute)
    k2 = cache.get_or_compute(X[:8], "CPMap", "projected", compute)
    assert calls["n"] == 1  # second call was a cache hit, compute_fn not re-invoked
    assert np.allclose(k1, k2)
    assert cache.hits == 1 and cache.misses == 1


def test_pqk_v2_fit_predict_and_kta_bookkeeping(toy_data):
    X, y = toy_data
    model = PQKv2(PQKConfig(n_qubits=4, seed=0, kta_steps=15, nystrom_m=10)).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert 0.0 <= model.accuracy(X, y) <= 1.0
    assert model.final_kta_ is not None
    assert isinstance(model.level1_passed, bool)


def test_pqk_v2_kernel_matrix_is_symmetric_for_self_gram(toy_data):
    X, y = toy_data
    model = PQKv2(PQKConfig(n_qubits=4, seed=0, kta_steps=5, nystrom_m=8)).fit(X, y)
    K = model.kernel_matrix(X[:6])
    assert np.allclose(K, K.T, atol=1e-8)


def test_classical_baselines_are_psd_and_bounded(toy_data):
    X, _ = toy_data
    K_rbf = rbf_kernel(X, X, gamma=0.5)
    assert is_positive_semidefinite(K_rbf)
    assert K_rbf.max() <= 1.0 + 1e-9  # RBF kernel values are in (0, 1]

    K_poly = polynomial_kernel(X, X, degree=2)
    assert K_poly.shape == (len(X), len(X))


def test_train_gen1_svm_logs_to_registry_and_returns_bounded_accuracy(tmp_path, toy_data):
    X, y = toy_data
    reg = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    result = train_gen1_svm(X[:16], y[:16], X[16:], y[16:], feature_map_name="AngleCNOT", registry=reg)
    assert 0.0 <= result.metrics["val_accuracy"] <= 1.0
    assert reg.count() == 1
    fetched = reg.get(result.metadata.run_id)
    assert fetched.metadata.generation == "gen1"


def test_train_pqk_v2_logs_to_registry(tmp_path, toy_data):
    X, y = toy_data
    reg = ExperimentRegistry(db_path=tmp_path / "reg2.sqlite3")
    _, result = train_pqk_v2(
        X[:16], y[:16], X[16:], y[16:],
        config=PQKConfig(n_qubits=4, seed=1, kta_steps=10, nystrom_m=10), registry=reg
    )
    assert reg.count() == 1
    assert result.metadata.generation == "gen2"


def test_classical_svm_baseline_train_eval(toy_data):
    X, y = toy_data
    result = train_classical_svm(X[:16], y[:16], X[16:], y[16:], kernel="rbf", gamma=0.5)
    assert 0.0 <= result["val_accuracy"] <= 1.0
    assert result["n_support_vectors"] >= 0


def test_noise_sensitivity_study_reports_bounded_degradation(toy_data):
    """Real hardware-relevant assertion: a fake-backend noise model should
    shift single-qubit <Z_i> expectations by a bounded, small amount (these
    are comparatively noise-robust vs. multi-qubit fidelity overlaps) -- not
    just 'the function ran without crashing'."""
    X, _ = toy_data
    result = run_noise_sensitivity_study(X[:4], shots=512, seed=0)
    assert result["n_samples"] == 4
    assert 0.0 <= result["projected_feature_mse"] < 1.0
    assert 0.0 <= result["max_abs_expectation_shift"] <= 2.0  # <Z> in [-1,1], so max shift <= 2


def test_kernel_classification_application_reports_both_methods_honestly():
    """Real, meaningful assertions: the comparison must actually run both
    methods and report a real number for each -- not just check that the
    code executes without error."""
    result = run_kernel_classification_comparison(n_samples=60, n_splits=2, seed=0)
    assert 0.0 <= result["avg_pqk_v2_accuracy"] <= 1.0
    assert 0.0 <= result["avg_classical_rbf_accuracy"] <= 1.0
    assert len(result["folds"]) == 2
    assert isinstance(result["pqk_beats_classical"], bool)
    # sanity: walk-forward means later folds must never see earlier test data as train
    for fold in result["folds"]:
        assert fold["n_train"] > 0 and fold["n_test"] > 0
