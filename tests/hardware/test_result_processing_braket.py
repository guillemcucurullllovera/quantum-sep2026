"""hardware.result_processing extended to Braket result objects,
not just Qiskit/IBM counts -- one normalized NormalizedResult schema
regardless of source."""
from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from hardware.braket.backend_adapter import run_local
from hardware.result_processing import (
    NormalizedResult,
    braket_counts_to_qiskit_convention,
    normalize_braket_local_result,
    normalize_braket_task_result,
    to_experiment_metrics,
    to_qubit_marginals,
)


def test_dense_braket_counts_match_qiskit_convention_bit_for_bit():
    """A circuit where every qubit is deterministic (all |1>) makes the
    correct converted bitstring unambiguous: qiskit little-endian convention
    for a 3-qubit all-ones state is '111' regardless of ordering, but a
    circuit where only qubit 0 is flipped disambiguates ordering."""
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(2)
    out = run_local(qc, shots=500)
    result = normalize_braket_local_result(out)
    assert isinstance(result, NormalizedResult)
    assert set(result.probabilities.keys()) == {"101"}  # q2=1, q1=0, q0=1 -> "101"
    assert result.shots == 500
    assert result.most_likely_bitstring == "101"


def test_sparse_idle_qubit_braket_counts_are_zero_padded_correctly():
    """Only qubit 0 is ever gated in a 4-declared-qubit circuit -- Braket's
    own measurement_counts bitstring is SHORTER than n_qubits (see
    hardware.braket.backend_adapter.run_local's 'measured_qubits' docs);
    normalize_braket_local_result must rebuild the full 4-bit width with
    the idle qubits as '0', not just naively reverse a too-short string."""
    qc = QuantumCircuit(4)
    qc.x(0)
    out = run_local(qc, shots=300)
    assert len(next(iter(out["measurement_counts"]))) == 1  # braket's raw string is short
    result = normalize_braket_local_result(out)
    assert set(result.probabilities.keys()) == {"0001"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_braket_counts_to_qiskit_convention_helper_directly():
    converted = braket_counts_to_qiskit_convention({"01": 10, "10": 5}, n_qubits=2)
    # braket '01' means qubit0=0,qubit1=1 -> qiskit little-endian "10"
    # braket '10' means qubit0=1,qubit1=0 -> qiskit little-endian "01"
    assert converted == {"10": 10, "01": 5}


def test_normalize_braket_local_result_rejects_analytic_mode_output():
    qc = QuantumCircuit(1)
    qc.h(0)
    out = run_local(qc, shots=0)  # analytic mode, no measurement_counts
    with pytest.raises(ValueError):
        normalize_braket_local_result(out)


def test_marginals_and_metrics_work_identically_on_normalized_braket_result():
    """The whole point of normalization: to_qubit_marginals/
    to_experiment_metrics (originally IBM-only) must work
    unmodified against a Braket-sourced, normalized counts dict."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    out = run_local(qc, shots=4000)
    result = normalize_braket_local_result(out)
    converted_counts = braket_counts_to_qiskit_convention(
        out["measurement_counts"], n_qubits=out["n_qubits"], measured_qubits=out.get("measured_qubits")
    )
    marginals = to_qubit_marginals(converted_counts)
    assert marginals[0] == pytest.approx(1.0, abs=1e-6)  # qubit 0 always |1>
    assert marginals[1] == pytest.approx(0.5, abs=0.06)  # qubit 1 ~coin flip

    metrics = to_experiment_metrics(result)
    assert metrics["shots"] == 4000.0
    assert metrics["num_distinct_outcomes"] in (2.0,)


def test_normalize_braket_task_result_handles_raw_sdk_result_object():
    """Not just this platform's run_local wrapper dict -- a raw Braket SDK
    result object (task.result()) too, per scope."""
    from hardware.braket.backend_adapter import qiskit_to_braket
    from braket.devices import LocalSimulator

    qc = QuantumCircuit(2)
    qc.x(0)
    translation = qiskit_to_braket(qc)
    device = LocalSimulator()
    task_result = device.run(translation.braket_circuit, shots=200).result()

    normalized = normalize_braket_task_result(task_result, n_qubits=2, backend_name="braket_raw")
    assert normalized.backend_name == "braket_raw"
    assert normalized.shots == 200
    assert set(normalized.probabilities.keys()) == {"01"}
