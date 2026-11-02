"""QAE state-preparation circuits (qiskit-native): A, S0, Grover Q, and Q^k A|000>.

Construction (n=2, 3 qubits: q0,q1 index; q2 ancilla). The ancilla rotation conditioned on
the index bits (b0,b1) must equal the multilinear angle

    theta(b0,b1) = c0 + c1*b0 + c2*b1 + c12*b0*b1,   theta_ij = 2*arcsin(sqrt(g(x_ij))).

Because RY rotations about a common axis ADD, we realise this exactly with
`RY(c0)` + `CRY(c1)[q0->q2]` + `CRY(c2)[q1->q2]` + `CCRY(c12)[q0,q1->q2]`, giving
`P(anc=1) = (1/4)*sum sin^2(theta/2) = mean_i g(x_i)` BY CONSTRUCTION (verified by
statevector in tests).

Adapted (MIT) from spinqit-qae-triangulum; state-prep rebuilt from the RY-addition
identity, not copied. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

from algorithms_extended.qae_spinqit.quadrature import Rule, grid_points, index_to_bits
from algorithms_extended.qae_spinqit.toy_integrands import Integrand, theta_from_value

Bits = Tuple[int, int]
AngleTable = Dict[Bits, float]

_TOL = 1e-12


def mobius_coeffs(t00: float, t01: float, t10: float, t11: float):
    """Moebius / multilinear expansion coefficients on B^2."""
    c0 = t00
    c1 = t10 - t00
    c2 = t01 - t00
    c12 = t11 - t10 - t01 + t00
    return c0, c1, c2, c12


def angle_table(integrand: Integrand, y: float = 1.0, rule: Rule = "midpoint", n: int = 2) -> AngleTable:
    """4-point (n=2) angle table theta_ij = 2*arcsin(sqrt(g(x_ij)))."""
    if n != 2:
        raise NotImplementedError("this pipeline targets n=2 (Triangulum/Kingston 3-qubit setup)")
    grid = grid_points(y=y, n=n, rule=rule)
    at: AngleTable = {}
    for i, x in enumerate(grid.points):
        at[index_to_bits(i, n)] = theta_from_value(integrand.g(x))
    return at


def build_A(at: AngleTable):
    """State-preparation operator A (no measurement)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate

    c0, c1, c2, c12 = mobius_coeffs(at[(0, 0)], at[(0, 1)], at[(1, 0)], at[(1, 1)])
    qc = QuantumCircuit(3, name="A")
    qc.h(0)
    qc.h(1)
    if abs(c0) > _TOL:
        qc.ry(c0, 2)
    if abs(c1) > _TOL:
        qc.cry(c1, 0, 2)
    if abs(c2) > _TOL:
        qc.cry(c2, 1, 2)
    if abs(c12) > _TOL:
        if abs(abs(c12) - math.pi) < 1e-9:
            qc.ccx(0, 1, 2)  # CCRy(+-pi) up to a phase that cancels in P(anc=1)
        else:
            qc.append(RYGate(c12).control(2), [0, 1, 2])
    return qc


def build_S0():
    """Reflection about |000>: I - 2|000><000| (up to global phase)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, name="S0")
    qc.x([0, 1, 2])
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x([0, 1, 2])
    return qc


def build_qae_circuit(at: AngleTable, k: int, measure: bool = True):
    """Q^k A |000> with Q = A * S0 * A^dagger * Z(anc)."""
    from qiskit import QuantumCircuit

    A = build_A(at)
    Adag = A.inverse()
    S0 = build_S0()

    qc = QuantumCircuit(3, 3, name=f"qae_k{k}")
    qc.compose(A, inplace=True)
    for _ in range(int(k)):
        qc.z(2)
        qc.compose(Adag, inplace=True)
        qc.compose(S0, inplace=True)
        qc.compose(A, inplace=True)
    if measure:
        qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _ancilla_prob(circuit) -> float:
    from qiskit.quantum_info import Statevector

    sv = Statevector.from_instruction(circuit)
    return sum(p for b, p in sv.probabilities_dict().items() if b[0] == "1")


def exact_amplitude(at: AngleTable) -> float:
    """Exact P(ancilla=1) of A|000> - should equal mean_i g(x_i)."""
    return _ancilla_prob(build_A(at))


def exact_pk(at: AngleTable, k: int) -> float:
    """Exact P(ancilla=1) of Q^k A|000> - the noiseless MLAE likelihood value p_k."""
    return _ancilla_prob(build_qae_circuit(at, k, measure=False))


def counts_ancilla_one(counts: Dict[str, int]) -> int:
    """Count shots with ancilla (q2, leftmost bit of the qiskit bitstring) = 1."""
    total = 0
    for b, c in counts.items():
        s = b.replace(" ", "")
        if s and s[0] == "1":
            total += c
    return total
