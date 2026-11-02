"""Noiseless-vs-noisy comparison on a real parameterized circuit, using
hardware.noise + hardware.fake_backends. Uses a small QAOA circuit
(algorithms.qaoa_qubo.solvers) as the representative test circuit rather than
a toy Bell state, since that's what this platform actually runs.
"""
from __future__ import annotations

import numpy as np
from qiskit_aer import AerSimulator

from algorithms.qaoa_qubo.solvers import _build_qiskit_qaoa_circuit
from hardware.fake_backends import resolve
from hardware.noise import models as noise_models
from hardware.transpilation.strategies import TranspilationStrategy, transpile_with_strategy


def _sample_qaoa_circuit(n: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    J = rng.uniform(-1, 1, size=(n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h = rng.uniform(-1, 1, size=n)
    gamma, beta = np.array([0.5]), np.array([0.3])
    circuit = _build_qiskit_qaoa_circuit(J, h, p=1, gamma=gamma, beta=beta)
    circuit.measure_all()
    return circuit


def noiseless_vs_noisy_bitstring_distribution(n_qubits: int = 4, shots: int = 4096, seed: int = 0):
    """Runs the same QAOA circuit noiseless (direct statevector-sampling
    simulation, no hardware-aware transpilation needed) vs. transpiled onto
    and run on a fake-Heron noise model, and reports the total-variation
    distance between the two output distributions -- the real, measurable
    cost of noise on this platform's actual circuits, not a synthetic toy.
    """
    circuit = _sample_qaoa_circuit(n_qubits, seed)
    resolved = resolve("fake_heron")
    noise_model, descriptor = noise_models.from_fake_backend(resolved.backend)

    sim_noiseless = AerSimulator()
    counts_noiseless = sim_noiseless.run(circuit, shots=shots, seed_simulator=seed).result().get_counts()

    transpiled_noisy = transpile_with_strategy(circuit, resolved.backend, TranspilationStrategy.MINIMIZE_DEPTH, seed=seed)
    sim_noisy = AerSimulator(noise_model=noise_model)
    counts_noisy = sim_noisy.run(transpiled_noisy.circuit, shots=shots, seed_simulator=seed).result().get_counts()

    tvd = total_variation_distance(counts_noiseless, counts_noisy, shots)
    return {
        "total_variation_distance": tvd,
        "noise_descriptor": descriptor.model_dump(),
        "transpiled_depth": transpiled_noisy.depth,
        "counts_noiseless": dict(counts_noiseless),
        "counts_noisy": dict(counts_noisy),
    }


def total_variation_distance(counts_a: dict, counts_b: dict, shots: int) -> float:
    keys = set(counts_a) | set(counts_b)
    return 0.5 * sum(abs(counts_a.get(k, 0) / shots - counts_b.get(k, 0) / shots) for k in keys)
