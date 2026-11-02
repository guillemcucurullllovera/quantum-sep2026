"""VQC ansatz library — pure-numpy exact statevector simulation.

Generalizes the v1-v5 progression into one parameterized implementation:

  VQCVersion.V1_LINEAR  — Gen1 v1-v3 style: single encoding layer, RY-only
                           variational block, linear CNOT chain. d=1.
  VQCVersion.V4_RING    — Gen1 v4 style: RY+RZ variational block, CNOT ring
                           entanglement, still single encoding. d=1.
  VQCVersion.V5_REUPLOAD— Gen2 v5 style: d=2 data re-uploading (each layer
                           re-encodes x), RY+RZ + CNOT ring per layer,
                           near-identity (sigma=0.01) initialization.

d (number of re-uploading layers) and entanglement topology are exposed as
config, not hardcoded, so future versions are additive rather than another
copy-pasted circuit_vN.py file.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.errors import ValidationError


class VQCVersion(str, Enum):
    V1_LINEAR = "v1_linear"
    V4_RING = "v4_ring"
    V5_REUPLOAD = "v5_reupload"


_VERSION_DEFAULTS = {
    VQCVersion.V1_LINEAR: dict(d_layers=1, entanglement="linear", use_rz=False, sigma_init=0.5),
    VQCVersion.V4_RING: dict(d_layers=1, entanglement="ring", use_rz=True, sigma_init=0.5),
    VQCVersion.V5_REUPLOAD: dict(d_layers=2, entanglement="ring", use_rz=True, sigma_init=0.01),
}


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(phi: float) -> np.ndarray:
    return np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])


def _axis_for_qubit(qubit: int, n: int) -> int:
    """Qiskit convention: qubit 0 is the least-significant bit of the basis
    index. A C-order reshape to [2]*n makes tensor axis 0 the MOST
    significant bit, so qubit `q` lives on tensor axis `n - 1 - q` — get this
    wrong and every gate still "runs", producing a normalized state that is
    silently permuted relative to any Qiskit-built circuit (caught by
    tests/parity/test_vqc_parity.py, which is exactly why that test exists)."""
    return n - 1 - qubit


def _apply_single_qubit_gate(state: np.ndarray, qubit: int, n: int, gate: np.ndarray) -> np.ndarray:
    axis = _axis_for_qubit(qubit, n)
    state_t = state.reshape([2] * n)
    state_t = np.tensordot(gate, state_t, axes=([1], [axis]))
    state_t = np.moveaxis(state_t, 0, axis)
    return state_t.reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    control_axis, target_axis = _axis_for_qubit(control, n), _axis_for_qubit(target, n)
    state_t = state.reshape([2] * n).copy()
    idx_c1_t0 = [slice(None)] * n
    idx_c1_t0[control_axis], idx_c1_t0[target_axis] = 1, 0
    idx_c1_t1 = [slice(None)] * n
    idx_c1_t1[control_axis], idx_c1_t1[target_axis] = 1, 1
    tmp = state_t[tuple(idx_c1_t0)].copy()
    state_t[tuple(idx_c1_t0)] = state_t[tuple(idx_c1_t1)]
    state_t[tuple(idx_c1_t1)] = tmp
    return state_t.reshape(-1)


def _entangle(state: np.ndarray, n: int, topology: str) -> np.ndarray:
    if topology == "linear":
        pairs = [(i, i + 1) for i in range(n - 1)]
    elif topology == "ring":
        pairs = [(i, (i + 1) % n) for i in range(n)]
    else:
        raise ValidationError(f"Unknown entanglement topology '{topology}'")
    for c, t in pairs:
        state = _apply_cnot(state, c, t, n)
    return state


@dataclass(frozen=True)
class VQCAnsatz:
    n_qubits: int
    version: VQCVersion = VQCVersion.V5_REUPLOAD
    d_layers: int | None = None
    entanglement: str | None = None
    use_rz: bool | None = None
    sigma_init: float | None = None

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValidationError("VQCAnsatz requires at least 2 qubits")
        defaults = _VERSION_DEFAULTS[self.version]
        object.__setattr__(self, "d_layers", self.d_layers or defaults["d_layers"])
        object.__setattr__(self, "entanglement", self.entanglement or defaults["entanglement"])
        object.__setattr__(self, "use_rz", self.use_rz if self.use_rz is not None else defaults["use_rz"])
        object.__setattr__(self, "sigma_init", self.sigma_init or defaults["sigma_init"])

    @property
    def params_per_layer(self) -> int:
        return self.n_qubits * (2 if self.use_rz else 1)

    @property
    def param_count(self) -> int:
        return self.d_layers * self.params_per_layer

    def init_params(self, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0, self.sigma_init, self.param_count)

    def _encoding_layer(self, state: np.ndarray, x: np.ndarray) -> np.ndarray:
        for i in range(self.n_qubits):
            angle = float(x[i]) if i < len(x) else 0.0
            state = _apply_single_qubit_gate(state, i, self.n_qubits, _ry(angle))
        return state

    def _trainable_layer(self, state: np.ndarray, layer_params: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        theta = layer_params[:n]
        for i in range(n):
            state = _apply_single_qubit_gate(state, i, n, _ry(float(theta[i])))
        if self.use_rz:
            phi = layer_params[n : 2 * n]
            for i in range(n):
                state = _apply_single_qubit_gate(state, i, n, _rz(float(phi[i])))
        return _entangle(state, n, self.entanglement)

    def statevector(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        if len(params) != self.param_count:
            raise ValidationError(
                f"Expected {self.param_count} params for {self.version.value} "
                f"(n={self.n_qubits}, d={self.d_layers}), got {len(params)}"
            )
        n = self.n_qubits
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0
        ppl = self.params_per_layer
        for layer in range(self.d_layers):
            state = self._encoding_layer(state, x)
            state = self._trainable_layer(state, params[layer * ppl : (layer + 1) * ppl])
        return state

    def z_expectations(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        state = self.statevector(x, params)
        probs = np.abs(state) ** 2
        basis = np.arange(len(state))
        n = self.n_qubits
        z_vals = np.zeros(n)
        for i in range(n):
            bits = (basis >> i) & 1
            z_vals[i] = float(np.dot(probs, 1 - 2 * bits))
        return z_vals

    def parameter_shift_feature_grad(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """d/dtheta_k of each z-expectation feature, shape (param_count, n_qubits).

        Exact for RY/RZ generators (eigenvalues +-1/2 => shift = pi/2 is exact).
        """
        shift = np.pi / 2
        grad = np.zeros((self.param_count, self.n_qubits))
        for k in range(self.param_count):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[k] += shift
            p_minus[k] -= shift
            grad[k] = (self.z_expectations(x, p_plus) - self.z_expectations(x, p_minus)) / 2.0
        return grad

    def gradient_variance_probe(self, n_samples: int = 50, seed: int = 42) -> float:
        """Barren-plateau diagnostic: Var[d<Z_j>/dtheta_k] over random (x, theta).

        This is the same diagnostic that justified the v1->v5 fixes (McClean
        2018 / Cerezo 2021): variance collapsing toward zero as n grows is the
        symptom; near-identity init is the mitigation this ansatz applies for
        V5_REUPLOAD by default.
        """
        rng = np.random.default_rng(seed)
        all_grads: list[float] = []
        for _ in range(n_samples):
            x = rng.uniform(-np.pi, np.pi, self.n_qubits)
            params = self.init_params(seed=int(rng.integers(0, 1_000_000)))
            all_grads.extend(self.parameter_shift_feature_grad(x, params).ravel().tolist())
        return float(np.var(all_grads))
