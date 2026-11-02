"""Real tests for algorithms_extended.qrbf_ocsvm_anomaly -- kernel correctness,
CV harness, honest documented-failure status, and the noisy hardware variant."""
from __future__ import annotations

import numpy as np
import pytest

from algorithms_extended.qrbf_ocsvm_anomaly import GATE_PASSED, MEASURED_TEST_F1, PAPER_TABLE_1
from algorithms_extended.qrbf_ocsvm_anomaly.kernels import (
    FEATURE_MAPS, N_QUBITS, compute_rdms_for_dataset, qrbf_kernel_matrix,
    rbf_kernel_matrix, scale_to_pi, state_to_1q_rdms,
)
from algorithms_extended.qrbf_ocsvm_anomaly.replication import (
    f1_score_anomaly, run_classical_cv, run_quantum_cv,
)
from core.metrics import is_positive_semidefinite


def test_documented_failure_status_is_honest():
    """The module must not silently claim success it hasn't earned."""
    assert GATE_PASSED is False
    for name in PAPER_TABLE_1:
        assert name in MEASURED_TEST_F1
        paper_mu, _ = PAPER_TABLE_1[name]
        measured_mu, _ = MEASURED_TEST_F1[name]
        # The documented finding is that quantum measured well BELOW paper
        # for every qrbf variant -- assert that's still what's recorded.
        if name != "rbf":
            assert measured_mu < paper_mu - 0.05, (
                f"{name}: measured {measured_mu} not documented as a clear miss "
                f"vs paper {paper_mu} -- did someone quietly soften this?"
            )


def test_1q_rdm_extraction_is_physical():
    """Each single-qubit RDM must be a valid density matrix: trace 1,
    Hermitian, positive semi-definite eigenvalues."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-np.pi, np.pi, size=2 * N_QUBITS)
    qc = FEATURE_MAPS["ECR"](x)
    from qiskit.quantum_info import Statevector
    sv = Statevector(qc).data
    rdms = state_to_1q_rdms(sv)
    assert rdms.shape == (N_QUBITS, 2, 2)
    for k in range(N_QUBITS):
        rho = rdms[k]
        assert abs(np.trace(rho).real - 1.0) < 1e-9
        assert np.allclose(rho, rho.conj().T, atol=1e-9)
        eigvals = np.linalg.eigvalsh(rho)
        assert (eigvals >= -1e-9).all()


def test_qrbf_kernel_matrix_is_psd_and_bounded():
    rng = np.random.default_rng(1)
    X = rng.uniform(-np.pi, np.pi, size=(12, 2 * N_QUBITS))
    rdms = compute_rdms_for_dataset(X, FEATURE_MAPS["CX"])
    K = qrbf_kernel_matrix(rdms, rdms, gamma=0.1)
    assert K.shape == (12, 12)
    assert np.allclose(np.diag(K), 1.0, atol=1e-9)  # K(x,x) = exp(0) = 1
    assert (K >= 0).all() and (K <= 1.0 + 1e-9).all()
    assert is_positive_semidefinite(K, tol=1e-6)


def test_classical_rbf_has_more_dynamic_range_than_qrbf_on_categorical_like_data():
    """Reproduces the REPLICATION_REPORT.md root-cause finding directly, on
    data structured like the real failure case: REPLICATION_REPORT.md
    section 3 measured this concentration effect on Sparkov's fraud-rate-
    encoded categorical features (few distinct values per column), not on
    generic continuous data.

    Verified empirically before writing this assertion: with fully
    continuous i.i.d. Gaussian features (no repeated-value structure), the
    concentration effect does NOT reliably reproduce -- the mechanism is
    specific to low-cardinality/repeated-value feature columns (like
    fraud-rate encoding), not an intrinsic property of the qrbf circuit for
    every input distribution. This test constructs categorical-like data
    (2-4 distinct levels per column, matching how the real dataset's
    fraud-rate-encoded features behave) to test the actual documented
    mechanism rather than a claim that doesn't hold in general.
    """
    rng = np.random.default_rng(2)
    n = 30
    X_raw = np.zeros((n, 2 * N_QUBITS))
    for col in range(2 * N_QUBITS):
        n_levels = rng.integers(2, 5)
        levels = rng.normal(size=n_levels)
        X_raw[:, col] = rng.choice(levels, size=n)
    X_scaled = scale_to_pi(X_raw)

    rdms = compute_rdms_for_dataset(X_scaled, FEATURE_MAPS["ECR"])
    K_q = qrbf_kernel_matrix(rdms, rdms, gamma=0.1)
    K_c = rbf_kernel_matrix(X_raw, X_raw, gamma=0.1)

    off_q = K_q[~np.eye(n, dtype=bool)]
    off_c = K_c[~np.eye(n, dtype=bool)]
    # The documented finding: quantum kernel's off-diagonal similarity is
    # concentrated near 1.0 (mean ~0.9, matching REPLICATION_REPORT.md's
    # measured 0.909) with a narrow range; classical RBF has much wider
    # dynamic range (matching the report's measured 9x-larger range).
    assert off_q.mean() > 0.8
    assert (off_q.max() - off_q.min()) < (off_c.max() - off_c.min())


def test_f1_score_anomaly_known_values():
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1])  # 1 TP, 1 FN, 1 FP
    f1 = f1_score_anomaly(y_true, y_pred)
    # precision = 1/2, recall = 1/2 -> F1 = 0.5
    assert abs(f1 - 0.5) < 1e-9


def test_f1_score_zero_true_positives_returns_zero():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert f1_score_anomaly(y_true, y_pred) == 0.0


def test_classical_cv_harness_runs_and_beats_chance_on_separable_toy_data():
    """Small, fast, real CV run (not the full paper-scale replication) --
    confirms the harness produces a sane, better-than-chance F1 on data with
    a genuine, well-separated anomaly signal.

    Threshold chosen from measurement, not assumption: swept gamma in
    {0.05,0.2,0.5,1.0} x nu in {0.15,0.2,0.25,0.3} on this exact toy case
    and found OCSVM's *unsupervised* training (it sees the anomalies mixed
    into the "normal" fit, unlike a supervised classifier) caps F1 around
    0.4-0.5 even for a 6-std-deviation mean shift -- nu only upper-bounds
    the assumed outlier fraction, so a mismatch between nu and the true
    contamination rate costs real F1 even when the classes are trivially
    separable in feature space. 0.35 is comfortably above the near-zero F1
    documented for the genuinely hard (kernel-concentrated) qrbf case, which
    is the actual comparison this test is guarding against regressing to."""
    rng = np.random.default_rng(3)
    normal = rng.normal(0, 1, size=(60, 5))
    anomaly = rng.normal(6, 1, size=(15, 5))  # well-separated, easy case
    X = np.vstack([normal, anomaly])
    y = np.concatenate([np.zeros(60, dtype=int), np.ones(15, dtype=int)])
    result = run_classical_cv(X, y, gamma=0.5, nu=0.25, k=5, seed=0)
    assert result["test_mean"] > 0.35


@pytest.mark.slow
def test_quantum_cv_harness_runs_end_to_end():
    """Slow (real circuit simulation per sample): confirms the quantum CV
    path executes and returns well-formed output, on a small instance."""
    rng = np.random.default_rng(4)
    X_raw = rng.normal(size=(20, 2 * N_QUBITS))
    y = np.array([0] * 16 + [1] * 4)
    X_scaled = scale_to_pi(X_raw)
    result = run_quantum_cv(X_scaled, y, "CX", gamma=0.1, nu=0.2, k=4, seed=0)
    assert 0.0 <= result["test_mean"] <= 1.0
    assert len(result["per_fold_test"]) == 4


def test_noisy_hardware_variant_rdms_are_physical():
    """The noisy-backend path must still produce valid density matrices
    (trace 1, Hermitian, PSD) -- noise should degrade purity, not break
    physicality."""
    from algorithms_extended.qrbf_ocsvm_anomaly.hardware_variant import compute_noisy_rdms_for_dataset

    rng = np.random.default_rng(5)
    X = rng.uniform(-np.pi, np.pi, size=(3, 2 * N_QUBITS))
    rdms, backend_name = compute_noisy_rdms_for_dataset(X, FEATURE_MAPS["CX"])
    assert rdms.shape == (3, N_QUBITS, 2, 2)
    assert isinstance(backend_name, str) and backend_name
    for i in range(3):
        for k in range(N_QUBITS):
            rho = rdms[i, k]
            assert abs(np.trace(rho).real - 1.0) < 1e-6
            assert np.allclose(rho, rho.conj().T, atol=1e-6)
            eigvals = np.linalg.eigvalsh(rho)
            assert (eigvals >= -1e-6).all()


def test_noise_does_not_trivially_fix_kernel_concentration():
    """Real diagnostic, not assumed: check whether noise changes the
    concentration finding. Either outcome is a legitimate result; the test
    only asserts the diagnostic runs and returns sane numbers, honestly
    reporting whichever direction it goes (see test output / benchmark logs
    for the actual comparison, this just guards against a crash or NaN)."""
    from algorithms_extended.qrbf_ocsvm_anomaly.hardware_variant import (
        compute_noisy_rdms_for_dataset, kernel_concentration,
    )

    rng = np.random.default_rng(6)
    X = rng.uniform(-np.pi, np.pi, size=(8, 2 * N_QUBITS))
    rdms, _ = compute_noisy_rdms_for_dataset(X, FEATURE_MAPS["CX"])
    stats = kernel_concentration(rdms)
    assert 0.0 <= stats["min_off_diag"] <= stats["mean_off_diag"] <= stats["max_off_diag"] <= 1.0 + 1e-6
    assert not any(np.isnan(v) for v in stats.values())
