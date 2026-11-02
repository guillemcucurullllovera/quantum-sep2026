from pytket.circuit import Circuit

from algorithms_extended.tket_asymmetrica.backends import get_detection_circuit


def test_detection_circuit_adds_leakage_detection_qubits_when_spare_capacity_exists():
    # get_detection_circuit only instruments *measured* qubits at the end of
    # the circuit -- an unmeasured circuit has nothing for it to detect.
    circuit = Circuit(3, 3).H(0).CX(0, 1).CX(1, 2)
    circuit.Measure(0, 0).Measure(1, 1).Measure(2, 2)

    gadget_circuit = get_detection_circuit(circuit, n_device_qubits=6)
    # with 3 spare device qubits available, the gadget must add postselection
    # ancillas rather than reusing measured data qubits
    assert gadget_circuit.n_qubits > circuit.n_qubits
    assert gadget_circuit.n_qubits <= 6


def test_detection_circuit_reuses_qubits_with_no_spare_capacity():
    circuit = Circuit(3, 3).H(0).CX(0, 1).CX(1, 2)
    circuit.Measure(0, 0).Measure(1, 1).Measure(2, 2)

    gadget_circuit = get_detection_circuit(circuit, n_device_qubits=3)
    # no spare device qubits: the gadget must reuse existing data qubits as
    # postselection ancillas rather than exceeding the device's qubit budget
    assert gadget_circuit.n_qubits == circuit.n_qubits
