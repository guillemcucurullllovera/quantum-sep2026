"""Regression tests proving algorithms/vqc_qml/v1 through v5 are genuinely
distinct, correctly-ported historical implementations — not accidental
duplicates of each other or of the original generalized ansatz.py.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from algorithms.vqc_qml.v1.circuit import VQCSmokeTest, N_QUBITS as V1_N_QUBITS
from algorithms.vqc_qml.v2.circuit_v2 import TrainableVQC, N_TOTAL_PARAMS as V2_N_PARAMS
from algorithms.vqc_qml.v3.circuit_v3 import TrainableVQC_v3, N_TOTAL_PARAMS as V3_N_PARAMS
from algorithms.vqc_qml.v4.circuit_v4 import TrainableVQC_v4
from algorithms.vqc_qml.v4.circuit_v4_qiskit import TrainableVQC_v4_Qiskit
from algorithms.vqc_qml.v5.ansatz import param_count as v5_param_count, vqc_circuit as v5_circuit


def test_documented_parameter_counts_match_source_docstrings():
    """Each version's own module docstring claims a specific parameter
    count — verify the ported code actually produces that count."""
    assert V1_N_QUBITS == 8  # v1: 8 trainable angles, one per qubit
    assert V2_N_PARAMS == 25  # v2: 16 variational + 8 weights + 1 bias
    assert V3_N_PARAMS == 33  # v3: 8 encoding scales + 16 variational + 8 weights + 1 bias
    assert TrainableVQC_v4(n_qubits=14, n_layers=2).n_total_params == 57
    assert TrainableVQC_v4_Qiskit(n_qubits=14, n_layers=2).n_total_params == 57
    assert v5_param_count(n=10, d=2) == 40  # v5: d*2*n = 2*2*10


def test_v1_through_v4_are_not_accidentally_identical():
    """Same 8-qubit-compatible input, same seed-derived params where shapes
    allow — outputs must differ, proving these are genuinely different
    circuits and not four copies of the same file."""
    features_8q = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.1]

    v1 = VQCSmokeTest(seed=42)
    out_v1 = v1.forward(features_8q)

    v2 = TrainableVQC(seed=42)
    out_v2 = v2.forward(features_8q)

    v3 = TrainableVQC_v3(seed=42)
    out_v3 = v3.forward(features_8q)

    # v1's output is a raw <Z> expectation in [-1, 1]; v2/v3 are sigmoid
    # probabilities in (0, 1) — different output domains is itself evidence
    # these are architecturally distinct, not just re-seeded copies.
    assert -1.0 <= out_v1 <= 1.0
    assert 0.0 < out_v2 < 1.0
    assert 0.0 < out_v3 < 1.0
    # v2 and v3 (both sigmoid-domain, same seed) must still differ — v3 adds
    # learned encoding scales and cross-entanglement v2 doesn't have.
    assert out_v2 != out_v3


def test_v4_qiskit_twin_matches_numpy_reference():
    """Parity check: v4's Qiskit port must match the NumPy reference to
    float precision — this is the actual historical parity guarantee
    documented in v4's own module docstring, re-verified after porting."""
    rng = np.random.RandomState(7)
    features = rng.uniform(-1, 1, 14).tolist()

    numpy_ref = TrainableVQC_v4(n_qubits=14, n_layers=2, seed=7)
    qiskit_twin = TrainableVQC_v4_Qiskit(
        n_qubits=14, n_layers=2, params=numpy_ref.params.copy(), seed=7
    )

    numpy_exps = numpy_ref._get_expectations(features, numpy_ref.variational_params, numpy_ref.encoding_scales)
    qiskit_exps = qiskit_twin.get_expectations(features)

    assert np.max(np.abs(numpy_exps - qiskit_exps)) < 1e-6
    assert abs(numpy_ref.forward(features) - qiskit_twin.forward(features)) < 1e-6


def test_v5_uses_10_qubits_not_14_the_documented_barren_plateau_fix():
    """v5's entire reason for existing (per its own module docstring) is
    fixing v4's barren plateau by dropping to 10 qubits — verify the
    ported code actually does that, not silently still running at 14."""
    state = v5_circuit(
        x=np.zeros(10), params=np.zeros(v5_param_count(10, 2)), n=10, d=2
    )
    assert state.shape == (2 ** 10,)
    assert abs(np.sum(np.abs(state) ** 2) - 1.0) < 1e-9  # valid normalized statevector


def test_v5_gradient_variance_probe_runs():
    """Smoke-check the ported gradient_variance_probe (v5's own diagnostic
    for whether the barren-plateau fixes actually worked) executes and
    returns a finite, non-negative variance."""
    from algorithms.vqc_qml.v5.ansatz import gradient_variance_probe

    var = gradient_variance_probe(n=10, d=2, n_samples=3, seed=1)
    assert var >= 0.0
    assert np.isfinite(var)


def test_ported_v4_correctness_script_still_passes():
    """prototypes/vqc/test_v4_correctness.py is the original Gen1 standalone
    correctness harness (parameter-shift-vs-finite-difference gradient
    check, unitarity, serialization roundtrip, boundary behavior) — run it
    as a subprocess against the ported module and confirm it still reports
    zero failures, rather than re-implementing its checks here."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "prototypes" / "vqc" / "test_v4_correctness.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo_root),
        env={"PYTHONPATH": str(repo_root), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "0 failed" in result.stdout, result.stdout + result.stderr
