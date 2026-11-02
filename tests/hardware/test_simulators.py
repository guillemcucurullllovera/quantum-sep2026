from qiskit import QuantumCircuit

from hardware.simulators import SimulatorMethod, build_simulator, recommend_method


def test_recommend_stabilizer_for_clifford_only_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    assert recommend_method(qc) == SimulatorMethod.STABILIZER


def test_recommend_statevector_for_small_noiseless_nonclifford():
    qc = QuantumCircuit(3)
    qc.rx(0.37, 0)
    qc.cx(0, 1)
    assert recommend_method(qc, has_noise_model=False) == SimulatorMethod.STATEVECTOR


def test_recommend_density_matrix_for_small_noisy_nonclifford():
    qc = QuantumCircuit(3)
    qc.rx(0.37, 0)
    assert recommend_method(qc, has_noise_model=True) == SimulatorMethod.DENSITY_MATRIX


def test_recommend_mps_for_large_circuit():
    qc = QuantumCircuit(30)
    qc.rx(0.1, 0)
    assert recommend_method(qc, has_noise_model=False, max_statevector_qubits=24) == SimulatorMethod.MATRIX_PRODUCT_STATE


def test_build_simulator_uses_requested_method():
    sim = build_simulator(SimulatorMethod.STATEVECTOR)
    assert sim.options.method == "statevector"


def test_build_simulator_and_actually_run_a_circuit():
    from qiskit import transpile

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    sim = build_simulator(SimulatorMethod.STATEVECTOR)
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=256).result()
    counts = result.get_counts()
    # a real Bell-state check: only 00/11 should appear (up to sampling noise, none here)
    assert set(counts.keys()) <= {"00", "11"}
