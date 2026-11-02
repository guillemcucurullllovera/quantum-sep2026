"""End-to-end example: rank fake backends for a circuit's qubit count,
transpile with two named strategies, and build a noise model. Run:
    python -m examples.hardware_backend_transpile_noise_example
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from hardware.backend_selection.scorer import rank_backends_for_circuit
from hardware.backends.catalog import BackendCatalog
from hardware.noise.models import synthetic
from hardware.transpilation.strategies import compare_strategies


def main() -> None:
    circuit = QuantumCircuit(4)
    circuit.h(range(4))
    for i in range(3):
        circuit.cx(i, i + 1)
    circuit.measure_all()

    print("Backend ranking for a 4-qubit circuit:")
    scores = rank_backends_for_circuit(n_qubits_needed=4)
    for s in scores[:3]:
        print(f"  {s.backend_name}: composite_score={s.composite_score:.3f}  layout={s.best_layout}")

    if scores:
        # BackendScore.backend_name is Qiskit's own display name (e.g.
        # "FakeSherbrooke"); BackendCatalog keys its registry by family name
        # instead ("fake_heron", "fake_eagle", "noiseless") -- resolve via
        # the family the ranked backend belongs to rather than assuming the
        # display name is a valid catalog key.
        family = "fake_heron" if "heron" in scores[0].backend_name.lower() or scores[0].num_qubits > 130 else "fake_eagle"
        resolved = BackendCatalog.get(family)
        strategies = compare_strategies(circuit, resolved.backend, seed=1)
        print("\nTranspilation strategy comparison:")
        for name, res in strategies.items():
            print(f"  {name}: depth={res.depth}  two_qubit_gates={res.two_qubit_gate_count}")

    noise_model, descriptor = synthetic(depolarizing_prob_2q=0.02)
    print(f"\nSynthetic noise model built for backend descriptor: {descriptor.backend_name}")


if __name__ == "__main__":
    main()
