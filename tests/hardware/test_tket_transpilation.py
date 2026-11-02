import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from hardware.tket_transpilation import (
    compare_three_strategies,
    qiskit_roundtrip,
    transpile_via_tket,
)


def _sample_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(0.7, 1)
    qc.cx(0, 3)
    qc.h(2)
    return qc


def test_roundtrip_preserves_statevector():
    qc = _sample_circuit()
    original_sv = Statevector.from_instruction(qc)
    roundtripped = qiskit_roundtrip(qc)
    roundtrip_sv = Statevector.from_instruction(roundtripped)
    # equal up to global phase
    fidelity = np.abs(np.vdot(original_sv.data, roundtrip_sv.data)) ** 2
    assert fidelity > 1 - 1e-9


def test_transpile_via_tket_preserves_unitary_on_optimizable_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(0)  # cancels
    qc.cx(0, 1)
    qc.x(1)
    qc.x(1)  # cancels
    original_sv = Statevector.from_instruction(qc)

    result = transpile_via_tket(qc)
    optimized_sv = Statevector.from_instruction(result.circuit)
    fidelity = np.abs(np.vdot(original_sv.data, optimized_sv.data)) ** 2
    assert fidelity > 1 - 1e-9
    # FullPeepholeOptimise should have cancelled the redundant H/X pairs
    assert result.circuit.size() < qc.size()


def test_zero_qubit_circuit_rejected():
    with pytest.raises(Exception):
        transpile_via_tket(QuantumCircuit(0))


def test_compare_three_strategies_produces_real_distinct_numbers():
    from hardware.backends.catalog import BackendCatalog

    names = BackendCatalog.list_backends()
    assert names, "expected at least one registered backend"
    resolved = BackendCatalog.get(names[0])

    qc = _sample_circuit()
    results = compare_three_strategies(qc, resolved.backend)

    assert set(results) == {"minimize_depth", "minimize_two_qubit_gates", "tket_full_peephole"}
    for name, r in results.items():
        assert r["depth"] > 0, f"{name} produced a zero-depth circuit"
        assert r["two_qubit_gate_count"] >= 3, f"{name} lost 2Q gates ({r})"
    # not all three should be identical -- otherwise the comparison is fake
    depths = {r["depth"] for r in results.values()}
    assert len(depths) > 1, f"all three strategies produced identical depth: {results}"
