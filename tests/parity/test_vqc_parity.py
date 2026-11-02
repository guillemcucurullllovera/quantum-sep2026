"""Parity test: our pure-numpy VQC statevector simulation must match an
equivalent circuit built and simulated in Qiskit, within tight tolerance —
the same validation discipline Gen1 applied between its NumPy and Qiskit
implementations (phase0/quantum_signal vs phase0/quantum_qiskit).
"""
import numpy as np
import pytest

from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _build_qiskit_circuit(x: np.ndarray, params: np.ndarray, n: int, d: int, use_rz: bool) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    ppl = n * (2 if use_rz else 1)
    for layer in range(d):
        for i in range(n):
            angle = float(x[i]) if i < len(x) else 0.0
            qc.ry(angle, i)
        layer_params = params[layer * ppl : (layer + 1) * ppl]
        theta = layer_params[:n]
        for i in range(n):
            qc.ry(float(theta[i]), i)
        if use_rz:
            phi = layer_params[n : 2 * n]
            for i in range(n):
                qc.rz(float(phi[i]), i)
        for i in range(n):
            qc.cx(i, (i + 1) % n)
    return qc


@pytest.mark.parametrize("version", [VQCVersion.V1_LINEAR, VQCVersion.V4_RING, VQCVersion.V5_REUPLOAD])
def test_statevector_parity_numpy_vs_qiskit(version):
    n = 4
    ansatz = VQCAnsatz(n_qubits=n, version=version)
    # V1_LINEAR uses linear entanglement in our implementation but the Qiskit
    # reference circuit here always uses a ring — restrict the parity check to
    # ring-topology versions where the two circuits are apples-to-apples.
    if ansatz.entanglement != "ring":
        pytest.skip("qiskit reference circuit in this test uses ring entanglement only")

    rng = np.random.default_rng(0)
    x = rng.uniform(-np.pi, np.pi, n)
    params = ansatz.init_params(seed=1)

    our_state = ansatz.statevector(x, params)

    qc = _build_qiskit_circuit(x, params, n, ansatz.d_layers, ansatz.use_rz)
    qiskit_state = Statevector.from_instruction(qc).data

    # global phase can differ; compare via fidelity instead of raw amplitudes
    fidelity = abs(np.vdot(our_state, qiskit_state)) ** 2
    assert fidelity > 1 - 1e-9


def test_z_expectation_parity_numpy_vs_qiskit():
    n = 3
    ansatz = VQCAnsatz(n_qubits=n, version=VQCVersion.V4_RING)
    rng = np.random.default_rng(2)
    x = rng.uniform(-np.pi, np.pi, n)
    params = ansatz.init_params(seed=2)

    our_z = ansatz.z_expectations(x, params)

    qc = _build_qiskit_circuit(x, params, n, ansatz.d_layers, ansatz.use_rz)
    sv = Statevector.from_instruction(qc)
    from qiskit.quantum_info import SparsePauliOp

    qiskit_z = np.array(
        [sv.expectation_value(SparsePauliOp("I" * (n - 1 - i) + "Z" + "I" * i)).real for i in range(n)]
    )

    assert np.allclose(our_z, qiskit_z, atol=1e-8)
