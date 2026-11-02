"""Noiseless vs. noisy vs. noisy-with-mitigation, three-way comparison, using
hardware.noise + hardware.zne + hardware.readout_mitigation. Two
independent real demonstrations: ZNE on an expectation value (gate-error
mitigation) and M3 on a GHZ-state readout distribution (readout-error
mitigation) -- deliberately different circuits, since ZNE needs an
unmeasured circuit to fold and M3 specifically targets readout, so
conflating them into one circuit would muddy which technique is doing what.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from hardware.fake_backends import resolve
from hardware.noise import models as noise_models
from hardware.readout_mitigation.m3 import M3Mitigator
from hardware.zne.zne import ZNEMitigator, fold_circuit_gates


def _prob_zero_on_qubit0(counts: dict, shots: int) -> float:
    n0 = sum(c for bitstring, c in counts.items() if bitstring[-1] == "0")
    return n0 / shots


def zne_expectation_mitigation(n_qubits: int = 3, shots: int = 4096, seed: int = 0):
    """ZNE on P(qubit0 = 0) for a small entangling circuit: noiseless exact
    value vs. raw-noisy vs. ZNE-extrapolated, on a fake-Heron noise model.
    """
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    circuit.rz(0.3, range(n_qubits))

    # noiseless exact reference
    ideal_circuit = circuit.copy()
    ideal_circuit.measure_all()
    exact_counts = AerSimulator().run(ideal_circuit, shots=shots, seed_simulator=seed).result().get_counts()
    exact_value = _prob_zero_on_qubit0(exact_counts, shots)

    resolved = resolve("fake_heron")
    noise_model, _ = noise_models.from_fake_backend(resolved.backend)
    sim_noisy = AerSimulator(noise_model=noise_model)

    def run_at_scale(scale_factor: int) -> float:
        folded = fold_circuit_gates(circuit, scale_factor)
        folded.measure_all()
        counts = sim_noisy.run(folded, shots=shots, seed_simulator=seed).result().get_counts()
        return _prob_zero_on_qubit0(counts, shots)

    raw_noisy_value = run_at_scale(1)
    mitigator = ZNEMitigator(noise_factors=[1, 3, 5])
    zne_value = mitigator.mitigate(run_at_scale)

    return {
        "exact_value": exact_value,
        "raw_noisy_value": raw_noisy_value,
        "zne_mitigated_value": zne_value,
        "raw_error": abs(raw_noisy_value - exact_value),
        "zne_error": abs(zne_value - exact_value),
        "zne_improved": abs(zne_value - exact_value) < abs(raw_noisy_value - exact_value),
    }


def m3_readout_mitigation(n_qubits: int = 3, shots: int = 4096):
    """M3 on a GHZ state: without readout mitigation, a GHZ state's counts
    leak probability mass onto bitstrings that shouldn't be possible under
    ideal readout (e.g. '010' for a 3-qubit GHZ) purely from readout error;
    M3 should visibly reduce that leaked mass.
    """
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    circuit.measure_all()

    resolved = resolve("fake_heron")
    noise_model, _ = noise_models.from_fake_backend(resolved.backend)
    sim = AerSimulator(noise_model=noise_model)
    raw_counts = sim.run(circuit, shots=shots).result().get_counts()

    valid_bitstrings = {"0" * n_qubits, "1" * n_qubits}

    def leaked_mass(counts: dict, total: float) -> float:
        return sum(c for bs, c in counts.items() if bs not in valid_bitstrings) / total

    raw_leak = leaked_mass(raw_counts, shots)

    mitigator = M3Mitigator(resolved.backend, qubits=list(range(n_qubits)))
    result = mitigator.apply(raw_counts)
    mitigated_leak = leaked_mass(result.mitigated_quasi_probabilities, 1.0)

    return {
        "raw_leaked_mass": raw_leak,
        "mitigated_leaked_mass": mitigated_leak,
        "m3_improved": mitigated_leak < raw_leak,
    }
