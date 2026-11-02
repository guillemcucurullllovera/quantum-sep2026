import numpy as np
import pytest

from hardware.coupling_maps import CouplingGraph, heavy_hex_reference
from hardware.fake_backends import list_fake_backends, resolve
from hardware.layout import calibration_aware_transpile, layout_method_sweep
from hardware.result_processing import normalize_counts, to_experiment_metrics, to_qubit_marginals
from hardware.routing import routing_method_sweep


def test_list_fake_backends_returns_real_descriptors():
    descriptors = list_fake_backends()
    assert len(descriptors) >= 1
    for d in descriptors:
        assert d.num_qubits > 0
        assert d.kind.value == "fake"


def test_list_fake_backends_min_qubits_filter_is_real():
    all_backends = list_fake_backends()
    high_bar = list_fake_backends(min_qubits=10_000)  # no real backend has this many
    assert len(high_bar) < len(all_backends)
    assert high_bar == []


def test_resolve_unknown_family_raises():
    from core.errors import BackendUnavailableError

    with pytest.raises(BackendUnavailableError):
        resolve("not_a_real_family")


def test_heavy_hex_reference_every_vertex_degree_at_most_3():
    graph = heavy_hex_reference(rows=4, cols=4)
    for node in graph.nodes:
        assert graph.degree(node) <= 3, f"node {node} has degree {graph.degree(node)} > 3"


def test_heavy_hex_reference_is_connected_for_reasonable_size():
    graph = heavy_hex_reference(rows=3, cols=3)
    assert graph.is_connected()


def test_calibration_aware_transpile_respects_qubit_count():
    from qiskit import QuantumCircuit

    from hardware.fake_backends import resolve as resolve_fake

    resolved = resolve_fake("fake_heron")
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    result = calibration_aware_transpile(qc, resolved.backend)
    assert result.circuit.num_qubits == resolved.backend.num_qubits  # transpiled onto full device register
    assert result.depth > 0


def test_layout_method_sweep_produces_a_result_per_method():
    from qiskit import QuantumCircuit

    from hardware.fake_backends import resolve as resolve_fake

    resolved = resolve_fake("fake_heron")
    qc = QuantumCircuit(4)
    for i in range(3):
        qc.cx(i, i + 1)
    out = layout_method_sweep(qc, resolved.backend, methods=("trivial", "dense"))
    assert set(out.keys()) == {"trivial", "dense"}
    for v in out.values():
        assert v["depth"] > 0


def test_routing_method_sweep_produces_a_result_per_method():
    from qiskit import QuantumCircuit

    from hardware.fake_backends import resolve as resolve_fake

    resolved = resolve_fake("fake_heron")
    qc = QuantumCircuit(4)
    for i in range(3):
        qc.cx(i, i + 1)
    out = routing_method_sweep(qc, resolved.backend, methods=("sabre", "stochastic"))
    assert "sabre" in out
    assert "depth" in out["sabre"] or "error" in out["sabre"]


def test_normalize_counts_probabilities_sum_to_one():
    counts = {"00": 400, "01": 100, "10": 100, "11": 400}
    result = normalize_counts(counts, backend_name="test")
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-9
    assert result.most_likely_bitstring in ("00", "11")


def test_normalize_counts_rejects_empty():
    with pytest.raises(ValueError):
        normalize_counts({}, backend_name="test")


def test_qubit_marginals_respects_little_endian_convention():
    # 2-qubit bitstring "01": qubit 0 (rightmost char) = '1', qubit 1 (leftmost char) = '0'
    counts = {"01": 1000}
    marginals = to_qubit_marginals(counts)
    assert marginals[0] == pytest.approx(1.0)
    assert marginals[1] == pytest.approx(0.0)


def test_to_experiment_metrics_shape():
    result = normalize_counts({"0": 500, "1": 500}, backend_name="test")
    metrics = to_experiment_metrics(result)
    assert set(metrics.keys()) == {"shots", "most_likely_probability", "num_distinct_outcomes"}
    assert all(isinstance(v, float) for v in metrics.values())
