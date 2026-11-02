"""hardware.circuit_analysis.compatibility -- a real pre-flight
"will this circuit run on this backend" check, catching an incompatible
circuit/backend pair BEFORE any execution attempt, with a clear, specific
exception, rather than failing deep inside a simulator call."""
from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from hardware.backend_selection.capabilities import (
    capabilities_from_braket_local_simulator,
    capabilities_from_ibm_backend,
)
from hardware.backends.catalog import BackendCatalog
from hardware.circuit_analysis.compatibility import (
    IncompatibleCircuitError,
    check_compatibility,
    require_compatible,
)


def test_compatible_circuit_passes_both_ibm_and_braket():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rzz(0.3, 1, 2)
    qc.measure_all()

    ibm_caps = capabilities_from_ibm_backend(BackendCatalog.get("fake_eagle"))
    braket_caps = capabilities_from_braket_local_simulator()

    ibm_report = check_compatibility(qc, ibm_caps)
    braket_report = check_compatibility(qc, braket_caps)
    assert ibm_report.compatible is True
    assert braket_report.compatible is True
    assert ibm_report.reasons == []
    assert braket_report.reasons == []


def test_gate_outside_braket_vocabulary_is_rejected_with_clear_reason():
    """A Toffoli (CCX) is genuinely outside hardware.braket.backend_adapter's
    translation vocabulary -- the compatibility check must catch this
    BEFORE execution, not let it fail deep inside qiskit_to_braket."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ccx(0, 1, 2)

    braket_caps = capabilities_from_braket_local_simulator()
    report = check_compatibility(qc, braket_caps)
    assert report.compatible is False
    assert any("ccx" in r for r in report.reasons)

    # The same circuit must be reported compatible against IBM (which
    # transpiles arbitrary gates -- no fixed vocabulary rejection there).
    ibm_caps = capabilities_from_ibm_backend(BackendCatalog.get("fake_eagle"))
    assert check_compatibility(qc, ibm_caps).compatible is True


def test_too_many_qubits_is_rejected_for_both_clouds():
    qc = QuantumCircuit(200)
    qc.h(0)

    ibm_caps = capabilities_from_ibm_backend(BackendCatalog.get("fake_eagle"))  # 127 qubits
    braket_caps = capabilities_from_braket_local_simulator()  # ~25 qubit practical ceiling

    ibm_report = check_compatibility(qc, ibm_caps)
    braket_report = check_compatibility(qc, braket_caps)
    assert ibm_report.compatible is False
    assert braket_report.compatible is False
    assert any("200" in r for r in ibm_report.reasons)


def test_shots_exceeding_max_shots_is_rejected():
    ibm_caps = capabilities_from_ibm_backend(BackendCatalog.get("fake_eagle"))
    qc = QuantumCircuit(1)
    qc.h(0)
    report = check_compatibility(qc, ibm_caps, shots=10_000_000)
    assert report.compatible is False
    assert any("shots" in r for r in report.reasons)


def test_require_compatible_raises_specific_exception_before_execution():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    braket_caps = capabilities_from_braket_local_simulator()

    with pytest.raises(IncompatibleCircuitError) as exc_info:
        require_compatible(qc, braket_caps)

    # The exception carries the full report, not just a generic message.
    assert exc_info.value.report.compatible is False
    assert exc_info.value.report.backend_name == braket_caps.name
    assert "ccx" in str(exc_info.value)


def test_require_compatible_passes_through_for_compatible_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    braket_caps = capabilities_from_braket_local_simulator()
    report = require_compatible(qc, braket_caps)  # must not raise
    assert report.compatible is True


def test_mid_circuit_control_flow_rejected_on_braket_but_not_ibm():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    qc.measure(1, 1)

    braket_caps = capabilities_from_braket_local_simulator()
    ibm_caps = capabilities_from_ibm_backend(BackendCatalog.get("fake_eagle"))

    braket_report = check_compatibility(qc, braket_caps)
    assert braket_report.compatible is False
    assert any("mid-circuit" in r for r in braket_report.reasons)

    ibm_report = check_compatibility(qc, ibm_caps)
    assert ibm_report.compatible is True
