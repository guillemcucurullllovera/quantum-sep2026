"""Quantum feature maps for kernel methods.

Ported from Gen1 (`kernel.py`
+ `phase0/quantum_qiskit/kernel_qiskit.py`) with the NumPy and Qiskit twins
kept side by side as in the original, since their exact numerical parity
(≤1e-6) is itself a tested property (see tests/parity/test_kernel_parity.py).

Also exposes Gen2's angle-encoding-plus-CNOT-ring map (originally in
`quantum/components/pqk_v2/pqk_v2.py::angle_encode_cnot_ring`) under the
same `encode(features) -> statevector` interface, since Gen2 was built to
fix a kernel-concentration failure specifically in the ZZFeatureMap below
(see algorithms/pqk/kta_alignment.py docstring for the citation).

References carried over from Gen1:
  - Non-variational QKMs: arXiv:2604.07896
  - Benchmarking QKT: arXiv:2408.10274
  - Projected QK: arXiv:2505.14593
  - Resource-efficient QK (CPMap): arXiv:2507.03689
  - Quantum temporal kernels: arXiv:2312.01602
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

N_QUBITS = 8
DIM = 2 ** N_QUBITS


# ── NumPy statevector primitives (Gen1) ─────────────────────────────────

def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


def _hadamard() -> np.ndarray:
    s = 1.0 / math.sqrt(2)
    return np.array([[s, s], [s, -s]], dtype=complex)


def _apply_single(state: np.ndarray, gate: np.ndarray, qubit: int, n: int = N_QUBITS) -> np.ndarray:
    tensor = state.reshape([2] * n)
    tensor = np.tensordot(gate, tensor, axes=([1], [qubit]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int, n: int = N_QUBITS) -> np.ndarray:
    dim = 2 ** n
    new_state = state.copy()
    visited: set[int] = set()
    for idx in range(dim):
        if idx in visited:
            continue
        if (idx >> (n - 1 - control)) & 1:
            flipped = idx ^ (1 << (n - 1 - target))
            if flipped != idx and flipped not in visited:
                new_state[idx] = state[flipped]
                new_state[flipped] = state[idx]
                visited.add(idx)
                visited.add(flipped)
    return new_state


def _apply_cphase(state: np.ndarray, control: int, target: int, phase: float, n: int = N_QUBITS) -> np.ndarray:
    dim = 2 ** n
    new_state = state.copy()
    factor = np.exp(1j * phase)
    for idx in range(dim):
        c_bit = (idx >> (n - 1 - control)) & 1
        t_bit = (idx >> (n - 1 - target)) & 1
        if c_bit and t_bit:
            new_state[idx] *= factor
    return new_state


def _apply_rzz(state: np.ndarray, q1: int, q2: int, theta: float, n: int = N_QUBITS) -> np.ndarray:
    dim = 2 ** n
    new_state = state.copy()
    for idx in range(dim):
        b1 = (idx >> (n - 1 - q1)) & 1
        b2 = (idx >> (n - 1 - q2)) & 1
        parity = 1 - 2 * (b1 ^ b2)
        new_state[idx] *= np.exp(-1j * theta / 2 * parity)
    return new_state


def _cnot_ring(state: np.ndarray, n: int = N_QUBITS) -> np.ndarray:
    for i in range(n - 1):
        state = _apply_cnot(state, i, i + 1, n)
    return _apply_cnot(state, n - 1, 0, n)


def _init_state(n: int = N_QUBITS) -> np.ndarray:
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    return state


def expectation_z(state: np.ndarray, qubit: int, n: int = N_QUBITS) -> float:
    probs = np.abs(state) ** 2
    idx = np.arange(len(state))
    bits = (idx >> (n - 1 - qubit)) & 1
    return float(np.dot(probs, 1 - 2 * bits))


def project_state(state: np.ndarray, n: int = N_QUBITS) -> np.ndarray:
    return np.array([expectation_z(state, q, n) for q in range(n)])


# ── Gen1 feature maps (NumPy) ────────────────────────────────────────────

class AngleCNOTMap:
    """RY(x*pi) encoding + CNOT ring, 2 reps. VQC-v2-style encoding layer."""

    name = "AngleCNOT"
    reps = 2

    def encode(self, features: Sequence[float]) -> np.ndarray:
        state = _init_state()
        for _ in range(self.reps):
            for i in range(N_QUBITS):
                state = _apply_single(state, _ry(float(features[i]) * math.pi), i)
            state = _cnot_ring(state)
        return state


class ZZFeatureMap:
    """Havlicek et al. (2019) ZZFeatureMap: H -> RZ(x) -> RZZ(xi,xj), 2 reps.

    NOTE: this is the map identified (see kta_alignment.py) as suffering
    kernel concentration for this platform's data — kept for parity/history,
    prefer AngleCNOTRingV2 (Gen2) for new work.
    """

    name = "ZZFeatureMap"
    reps = 2

    def encode(self, features: Sequence[float]) -> np.ndarray:
        state = _init_state()
        H = _hadamard()
        for _ in range(self.reps):
            for i in range(N_QUBITS):
                state = _apply_single(state, H, i)
            for i in range(N_QUBITS):
                state = _apply_single(state, _rz(float(features[i]) * math.pi), i)
            for i in range(N_QUBITS - 1):
                theta_zz = (math.pi - float(features[i])) * (math.pi - float(features[i + 1]))
                state = _apply_rzz(state, i, i + 1, theta_zz)
        return state


class CPMap:
    """Resource-efficient map (Singh et al., arXiv:2507.03689): RY + CP chain."""

    name = "CPMap"
    reps = 2

    def encode(self, features: Sequence[float]) -> np.ndarray:
        state = _init_state()
        for _ in range(self.reps):
            for i in range(N_QUBITS):
                state = _apply_single(state, _ry(float(features[i]) * math.pi), i)
            for i in range(N_QUBITS - 1):
                phase = float(features[i]) * float(features[i + 1]) * math.pi
                state = _apply_cphase(state, i, i + 1, phase)
        return state


class AngleCNOTRingV2:
    """Gen2's fix for ZZFeatureMap's kernel concentration: RY(x*pi) encoding
    + a single CNOT ring (area-law entanglement), optionally with trainable
    per-qubit multiplicative encoding parameters (used by KTA alignment).
    Ported from `quantum/components/pqk_v2/pqk_v2.py::angle_encode_cnot_ring`.

    `n_qubits` is configurable (the original Gen2 test suite ran this at
    n_qubits=4 for speed) -- it must NOT be hardcoded to the module's
    N_QUBITS=8 constant, since PQKConfig.n_qubits needs to actually control
    circuit width, not be silently ignored.
    """

    name = "AngleCNOTRingV2"
    reps = 1

    def __init__(self, n_qubits: int = N_QUBITS) -> None:
        self.n_qubits = n_qubits

    def encode(self, features: Sequence[float], encoding_params: np.ndarray | None = None) -> np.ndarray:
        n = self.n_qubits
        x = np.asarray(features[:n], dtype=float)
        if encoding_params is not None:
            x = x * np.asarray(encoding_params[:n])
        state = _init_state(n)
        for i in range(n):
            angle = float(x[i]) * math.pi if i < len(x) else 0.0
            state = _apply_single(state, _ry(angle), i, n)
        for i in range(n):
            state = _apply_cnot(state, i, (i + 1) % n, n)
        return state


FEATURE_MAPS: dict[str, type] = {
    "AngleCNOT": AngleCNOTMap,
    "ZZFeatureMap": ZZFeatureMap,
    "CPMap": CPMap,
    "AngleCNOTRingV2": AngleCNOTRingV2,
}


# ── Qiskit twins (Gen1 parity set) ──────────────────────────────────────

def _z_expectations_qiskit(sv: Statevector, n: int = N_QUBITS) -> np.ndarray:
    probs = np.abs(sv.data) ** 2
    indices = np.arange(len(sv.data))
    return np.array([float(np.dot(probs, 1 - 2 * ((indices >> q) & 1))) for q in range(n)])


class AngleCNOTMap_Qiskit:
    name = "AngleCNOT"
    reps = 2

    def encode(self, features: Sequence[float]) -> Statevector:
        qc = QuantumCircuit(N_QUBITS)
        for _ in range(self.reps):
            for i in range(N_QUBITS):
                qc.ry(float(features[i]) * math.pi, i)
            for i in range(N_QUBITS - 1):
                qc.cx(i, i + 1)
            qc.cx(N_QUBITS - 1, 0)
        return Statevector.from_instruction(qc)


class ZZFeatureMap_Qiskit:
    name = "ZZFeatureMap"
    reps = 2

    def encode(self, features: Sequence[float]) -> Statevector:
        qc = QuantumCircuit(N_QUBITS)
        for _ in range(self.reps):
            for i in range(N_QUBITS):
                qc.h(i)
            for i in range(N_QUBITS):
                qc.rz(float(features[i]) * math.pi, i)
            for i in range(N_QUBITS - 1):
                theta_zz = (math.pi - float(features[i])) * (math.pi - float(features[i + 1]))
                qc.rzz(theta_zz, i, i + 1)
        return Statevector.from_instruction(qc)


class CPMap_Qiskit:
    name = "CPMap"
    reps = 2

    def encode(self, features: Sequence[float]) -> Statevector:
        qc = QuantumCircuit(N_QUBITS)
        for _ in range(self.reps):
            for i in range(N_QUBITS):
                qc.ry(float(features[i]) * math.pi, i)
            for i in range(N_QUBITS - 1):
                phase = float(features[i]) * float(features[i + 1]) * math.pi
                qc.cp(phase, i, i + 1)
        return Statevector.from_instruction(qc)


class AngleCNOTRingV2_Qiskit:
    name = "AngleCNOTRingV2"
    reps = 1

    def __init__(self, n_qubits: int = N_QUBITS) -> None:
        self.n_qubits = n_qubits

    def encode(self, features: Sequence[float], encoding_params: np.ndarray | None = None) -> Statevector:
        n = self.n_qubits
        x = np.asarray(features[:n], dtype=float)
        if encoding_params is not None:
            x = x * np.asarray(encoding_params[:n])
        qc = QuantumCircuit(n)
        for i in range(n):
            angle = float(x[i]) * math.pi if i < len(x) else 0.0
            qc.ry(angle, i)
        for i in range(n):
            qc.cx(i, (i + 1) % n)
        return Statevector.from_instruction(qc)


FEATURE_MAPS_QISKIT: dict[str, type] = {
    "AngleCNOT": AngleCNOTMap_Qiskit,
    "ZZFeatureMap": ZZFeatureMap_Qiskit,
    "CPMap": CPMap_Qiskit,
    "AngleCNOTRingV2": AngleCNOTRingV2_Qiskit,
}
