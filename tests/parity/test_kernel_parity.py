"""NumPy-vs-Qiskit parity for the Gen1 feature maps.

Carries forward the original parity guarantee documented in
the Gen1 Qiskit kernel implementation:
max|K_numpy - K_qiskit| <= 1e-6 for the fidelity kernel, and the same bound
for projected <Z_i> features -- despite NumPy's big-endian vs. Qiskit's
little-endian qubit ordering, because both quantities are invariant under
the same permutation applied to both operands (see that module's original
docstring for the argument).
"""
import numpy as np
import pytest

from algorithms.pqk.feature_maps import (
    FEATURE_MAPS,
    FEATURE_MAPS_QISKIT,
    N_QUBITS,
    project_state,
)


def _z_expectations_from_statevector_qiskit(sv, n=N_QUBITS):
    probs = np.abs(sv.data) ** 2
    indices = np.arange(len(sv.data))
    return np.array([float(np.dot(probs, 1 - 2 * ((indices >> q) & 1))) for q in range(n)])


@pytest.fixture()
def sample_features():
    rng = np.random.default_rng(1)
    return [rng.uniform(0, 1, size=N_QUBITS) for _ in range(4)]


@pytest.mark.parametrize("name", sorted(FEATURE_MAPS))
def test_projected_features_parity(name, sample_features):
    numpy_map = FEATURE_MAPS[name]()
    qiskit_map = FEATURE_MAPS_QISKIT[name]()

    for x in sample_features:
        state_numpy = numpy_map.encode(x)
        proj_numpy = project_state(state_numpy)

        sv_qiskit = qiskit_map.encode(x)
        proj_qiskit = _z_expectations_from_statevector_qiskit(sv_qiskit)

        max_diff = float(np.max(np.abs(proj_numpy - proj_qiskit)))
        assert max_diff <= 1e-6, f"{name}: projected-feature parity broken, max diff={max_diff}"


@pytest.mark.parametrize("name", sorted(FEATURE_MAPS))
def test_fidelity_kernel_parity(name, sample_features):
    numpy_map = FEATURE_MAPS[name]()
    qiskit_map = FEATURE_MAPS_QISKIT[name]()

    x1, x2 = sample_features[0], sample_features[1]

    state_numpy_1, state_numpy_2 = numpy_map.encode(x1), numpy_map.encode(x2)
    fidelity_numpy = float(abs(np.dot(state_numpy_1.conj(), state_numpy_2)) ** 2)

    sv1, sv2 = qiskit_map.encode(x1), qiskit_map.encode(x2)
    fidelity_qiskit = float(abs(sv1.inner(sv2)) ** 2)

    assert abs(fidelity_numpy - fidelity_qiskit) <= 1e-6, (
        f"{name}: fidelity-kernel parity broken, numpy={fidelity_numpy}, qiskit={fidelity_qiskit}"
    )
