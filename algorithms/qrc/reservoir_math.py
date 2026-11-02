"""Core quantum-reservoir-computing math: Hamiltonian, evolution, encoding,
partial trace, ridge readout.

Licensing note (see THIRD_PARTY_NOTICES.md): this module implements
a standard, widely-published quantum reservoir computing architecture —
a transverse-field Ising Hamiltonian reservoir with angle-encoded inputs,
partial-trace state recycling, and Z-expectation readout — the general
pattern established in the broader QRC literature (Fujii & Nakajima 2017,
"Harnessing disordered-ensemble quantum dynamics for machine learning",
and its many successors), not the exclusive contribution of any single
paper. It is a clean-room implementation (written from physics/equations,
never copied from any third-party source code) and contains NO code from
any unlicensed repository.

This code was originally written (as `algorithms_extended/qrc_arxiv_replication/
reservoir_core.py`) while replicating a specific paper (Li, Mukhopadhyay,
Bayat, Habibnia, arXiv:2505.13933) whose own reference implementation's
upstream GitHub repository was confirmed (live check) to carry no
LICENSE file and no license field at all — "no license granted" under
default copyright, not merely "unverified". Because these functions
implement a general, independently-derivable QRC architecture rather than
copying that repository's specific source code, and because they are also
the mathematical foundation this platform's own Gen2 `QuantumReservoir`
(built independently — see algorithms/qrc/reservoir.py's own
docstring for its git-verified provenance) depends on, they were relocated
here out of the isolated/excluded `algorithms_extended/qrc_arxiv_replication/`
package rather than excluded outright. The parts that remain genuinely and
specifically tied to reproducing-and-comparing-against that one paper's
published Table II numbers (the paper's own MSE table, the specific
replication experiment driver, and the detailed paper-comparison narrative)
stay isolated outside the publication tree — see
`_EXCLUDED_FROM_PUBLICATION/qrc_arxiv_replication_driver/`.

Asymmetrica's own from-scratch reimplementation using this exact math,
against this exact paper's target, did NOT pass its own internal validation
gate (documented honestly below) — that is Asymmetrica's own research
finding about our port, not paper content, and is preserved here.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

# -- Honest internal research finding (Asymmetrica's own, not paper content) --
# Best documented result of Asymmetrica's from-scratch reimplementation of
# this architecture, targeting a specific published benchmark, after a
# readout-calibration sweep. This is a fact about our port, not a target to
# chase by curve-fitting. See _EXCLUDED_FROM_PUBLICATION/qrc_arxiv_replication_driver/
# for the full comparison against the paper's published Table II.
ASYMMETRICA_REIMPLEMENTATION_BEST_MSE = 0.2066
GATE_THRESHOLD_MULTIPLE = 1.1  # phase-0 decision rule: must be <= 1.1x target MSE to pass
_TARGET_MSE_FOR_GATE = 0.105  # the specific published target this was checked against
GATE_PASSED = (ASYMMETRICA_REIMPLEMENTATION_BEST_MSE / _TARGET_MSE_FOR_GATE) <= GATE_THRESHOLD_MULTIPLE
assert GATE_PASSED is False, (
    "This constant records a historical fact (the gate was not met). If a "
    "future reimplementation actually passes the gate, update this module "
    "deliberately — don't let this assertion silently start failing."
)
ARXIV_GATE_NOTE = (
    f"QRC gen1_arxiv-style architecture: Asymmetrica's own reimplementation "
    f"reached MSE={ASYMMETRICA_REIMPLEMENTATION_BEST_MSE}, "
    f"{ASYMMETRICA_REIMPLEMENTATION_BEST_MSE / _TARGET_MSE_FOR_GATE:.2f}x the "
    f"internal validation gate's target — did not pass the phase-0 gate "
    f"(<=1.1x). Documented honestly, not hidden. See "
    f"THIRD_PARTY_NOTICES.md "
    f"for the full comparison history."
)

# -- Pauli matrices -----------------------------------------------------

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron_chain(mats: list[np.ndarray]) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def pauli_on(op: np.ndarray, i: int, n: int) -> np.ndarray:
    """Place a 2x2 Pauli `op` on qubit `i` of an n-qubit system; I elsewhere."""
    return _kron_chain([op if k == i else I2 for k in range(n)])


def two_pauli_on(op: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
    """Place `op` on both qubits i and j (i != j)."""
    return _kron_chain([op if k in (i, j) else I2 for k in range(n)])


# -- Hamiltonian ----------------------------------------------------------

def build_ising_hamiltonian(n: int, couplings: np.ndarray, field_nu: float = 1.0) -> np.ndarray:
    """Fully-connected transverse-field Ising: H = sum_{i<j} J_ij X_i X_j + nu sum_i Z_i.

    Standard QRC reservoir Hamiltonian (see module docstring for literature
    context) — not exclusive to any one paper.
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        for j in range(i + 1, n):
            H = H + couplings[i, j] * two_pauli_on(X, i, j, n)
    for i in range(n):
        H = H + field_nu * pauli_on(Z, i, n)
    return H


def random_couplings(n: int, rng: np.random.Generator) -> np.ndarray:
    """J_ij ~ U(0, 1), symmetric, zero diagonal.

    NOTE: unlike `algorithms/qrc/reservoir.py` (Gen2)'s own coupling
    generator, this does NOT normalize spectral radius. Kept unnormalized
    deliberately as the "gen1-style" variant for historical/comparative
    research purposes — see algorithms/qrc/reservoir.py for the
    spectral-radius-normalized fix and why it matters.
    """
    U = rng.random((n, n))
    U = np.triu(U, k=1)
    return U + U.T


def evolution_operator(H: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """U(tau) = exp(-i H tau)."""
    return expm(-1j * H * tau)


# -- R_Y encoding -----------------------------------------------------------

def ry_encoding_matrix(angles: np.ndarray) -> np.ndarray:
    """R_Y(x) = kron_j exp(-i (x_j / 2) Y_j) — standard angle encoding."""
    R = np.array([[1.0]], dtype=np.complex128)
    for a in angles:
        c, s = np.cos(a / 2.0), np.sin(a / 2.0)
        ry_single = np.array([[c, -s], [s, c]], dtype=np.complex128)
        R = np.kron(R, ry_single)
    return R


def encode_input_state(angles: np.ndarray) -> np.ndarray:
    """rho_I = R_Y(x) |0><0|^{n1} R_Y(x)^dagger."""
    R = ry_encoding_matrix(angles)
    first_col = R[:, 0:1]
    return first_col @ first_col.conj().T


# -- Partial trace ------------------------------------------------------

def partial_trace_input(rho: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Trace out the first n1 qubits, keep the last n2 (hidden) qubits."""
    d1, d2 = 1 << n1, 1 << n2
    rho_re = rho.reshape(d1, d2, d1, d2)
    return np.einsum("ijik->jk", rho_re)


def precompute_z_diagonals(n: int) -> np.ndarray:
    """Row j = diag(Z_j) in the computational basis, for fast Tr(rho Z_j)."""
    dim = 1 << n
    indices = np.arange(dim)
    diags = np.empty((n, dim), dtype=np.float64)
    for j in range(n):
        bit_j = (indices >> (n - 1 - j)) & 1
        diags[j] = 1.0 - 2.0 * bit_j
    return diags


# -- QRC forward pass -----------------------------------------------------

def qrc_forward(
    inputs: np.ndarray,
    U: np.ndarray,
    n1: int,
    n2: int,
    U_dag: np.ndarray | None = None,
    z_diags: np.ndarray | None = None,
) -> np.ndarray:
    """Run the 3-step windowed quantum reservoir; return the Z-expectation vector.

    `inputs`: shape (3, n1). Returns shape (n1+n2,) in [-1, 1].
    """
    n = n1 + n2
    dim_hidden = 1 << n2
    if U_dag is None:
        U_dag = U.conj().T
    if z_diags is None:
        z_diags = precompute_z_diagonals(n)

    rho_hidden = np.zeros((dim_hidden, dim_hidden), dtype=np.complex128)
    rho_hidden[0, 0] = 1.0

    n_steps = len(inputs)
    rho = None
    for step in range(n_steps):
        x = inputs[step]
        rho_input = encode_input_state(x)
        rho = np.kron(rho_input, rho_hidden)
        rho = U @ rho @ U_dag
        if step < n_steps - 1:
            rho_hidden = partial_trace_input(rho, n1, n2)

    diag = np.real(np.diag(rho))
    return z_diags @ diag


# -- Ridge regression readout (generic textbook math, no paper-specificity) --

def ridge_fit(M: np.ndarray, y: np.ndarray, delta: float = 1e-8, bias: bool = True) -> np.ndarray:
    """Solve W = (M M^T + delta I)^{-1} M y — standard L2-regularized linear
    readout. `bias=True` appends a constant row to M (un-regularized intercept).
    """
    if bias:
        M = np.vstack([M, np.ones((1, M.shape[1]))])
    n_features = M.shape[0]
    A = M @ M.T + delta * np.eye(n_features)
    b = M @ y
    return np.linalg.solve(A, b)


def ridge_predict(W: np.ndarray, M: np.ndarray, bias: bool = True) -> np.ndarray:
    """y_hat = W^T M (with the same bias-row convention as ridge_fit)."""
    if bias:
        M = np.vstack([M, np.ones((1, M.shape[1]))])
    return W @ M


__all__ = [
    "build_ising_hamiltonian",
    "random_couplings",
    "evolution_operator",
    "precompute_z_diagonals",
    "qrc_forward",
    "ridge_fit",
    "ridge_predict",
    "encode_input_state",
    "ry_encoding_matrix",
    "ASYMMETRICA_REIMPLEMENTATION_BEST_MSE",
    "GATE_THRESHOLD_MULTIPLE",
    "GATE_PASSED",
    "ARXIV_GATE_NOTE",
]
