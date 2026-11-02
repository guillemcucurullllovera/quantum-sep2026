"""hardware.backend_selection.capabilities -- BackendCapabilities
normalized across IBM fake backends and Braket's local simulator, so
application code can query "can this backend run this circuit" generically."""
from __future__ import annotations

from hardware.backend_selection.capabilities import (
    BackendCapabilities,
    capabilities_from_braket_local_simulator,
    capabilities_from_ibm_backend,
)
from hardware.backends.catalog import BackendCatalog


def test_ibm_fake_backend_capabilities_report_real_qubit_count_and_noise():
    resolved = BackendCatalog.get("fake_eagle")
    caps = capabilities_from_ibm_backend(resolved)
    assert isinstance(caps, BackendCapabilities)
    assert caps.provider == "ibm"
    assert caps.cloud == "ibm_quantum"
    assert caps.num_qubits == resolved.descriptor.num_qubits == 127
    assert caps.is_noisy is True
    assert caps.is_simulator is True  # FAKE is still a local simulation, not a real QPU
    assert caps.native_gates is None  # IBM path transpiles arbitrary gates, no fixed vocabulary
    assert caps.supports_mid_circuit_measurement is True
    assert caps.calibration_source == resolved.descriptor.calibration_source


def test_ibm_noiseless_simulator_capabilities_report_no_noise():
    resolved = BackendCatalog.get("noiseless")
    caps = capabilities_from_ibm_backend(resolved)
    assert caps.is_noisy is False
    assert caps.is_simulator is True


def test_braket_local_simulator_capabilities_have_a_fixed_gate_vocabulary():
    caps = capabilities_from_braket_local_simulator()
    assert caps.provider == "aws"
    assert caps.cloud == "aws_braket_local"
    assert caps.is_simulator is True
    assert caps.is_noisy is False
    # Unlike IBM, Braket's path really does reject gates outside a fixed set.
    assert caps.native_gates is not None
    assert "rzz" in caps.native_gates  # gate-vocabulary extension
    assert "cp" in caps.native_gates  # gate-vocabulary extension
    assert "ccx" not in caps.native_gates  # still genuinely unsupported
    # No mid-circuit measurement: qiskit_to_braket drops measurement entirely.
    assert caps.supports_mid_circuit_measurement is False


def test_capabilities_genuinely_differ_between_ibm_and_braket():
    """The actual point of normalization: the two clouds' capabilities
    must be comparable (same schema) AND must actually differ where they
    really do differ -- gate-vocabulary restriction and mid-circuit
    measurement support are not incidental fields, they're the two
    concrete asymmetries between this platform's IBM and Braket execution
    paths."""
    ibm_caps = capabilities_from_ibm_backend(BackendCatalog.get("fake_eagle"))
    braket_caps = capabilities_from_braket_local_simulator()

    assert ibm_caps.native_gates is None
    assert braket_caps.native_gates is not None

    assert ibm_caps.supports_mid_circuit_measurement is True
    assert braket_caps.supports_mid_circuit_measurement is False

    assert ibm_caps.cloud != braket_caps.cloud
    assert ibm_caps.max_shots != braket_caps.max_shots
