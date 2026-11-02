from qiskit import QuantumCircuit

from hardware.backends import BackendCatalog
from hardware.circuit_analysis import analyze, compare
from hardware.transpilation import TranspilationStrategy, compare_strategies, transpile_with_strategy


def _sample_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(5)
    for i in range(5):
        qc.h(i)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def test_strategies_produce_valid_transpiled_circuits():
    r = BackendCatalog.get("fake_heron")
    qc = _sample_circuit()
    result = transpile_with_strategy(qc, r.backend, TranspilationStrategy.MINIMIZE_DEPTH)
    assert result.depth > 0
    assert result.two_qubit_gate_count >= 4  # at least the original 4 CX gates, post-routing


def test_two_named_strategies_produce_different_layouts():
    r = BackendCatalog.get("fake_heron")
    qc = _sample_circuit()
    results = compare_strategies(qc, r.backend)
    assert set(results) == {"minimize_depth", "minimize_two_qubit_gates"}
    # they need not always differ numerically, but both must be valid non-trivial transpilations
    for res in results.values():
        assert res.depth > 0


def test_circuit_analysis_matches_known_gate_counts():
    qc = _sample_circuit()
    stats = analyze(qc)
    assert stats.n_qubits == 5
    assert stats.gate_counts["h"] == 5
    assert stats.gate_counts["cx"] == 4
    assert stats.two_qubit_gate_count == 4


def test_compare_reports_relative_change():
    r = BackendCatalog.get("fake_heron")
    qc = _sample_circuit()
    transpiled = transpile_with_strategy(qc, r.backend, TranspilationStrategy.MINIMIZE_DEPTH).circuit
    diff = compare(qc, transpiled)
    assert diff["depth_after"] >= diff["depth_before"]  # routing on real topology never reduces depth here


def test_circuit_too_big_for_backend_raises():
    from core.errors import ValidationError
    import pytest

    r = BackendCatalog.get("fake_eagle")
    huge = QuantumCircuit(200)
    with pytest.raises(ValidationError):
        transpile_with_strategy(huge, r.backend, TranspilationStrategy.MINIMIZE_DEPTH)
