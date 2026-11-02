"""Circuit-template construction logic, adapted from PennyLane's
``pennylane.templates`` (Apache-2.0, Xanadu Quantum Technologies Inc.).

See ``THIRD_PARTY_NOTICES.md`` (section) for the key insight this
subpackage relies on: a template's ``compute_decomposition`` staticmethod is
*pure gate-sequence-construction logic* -- given a weight tensor and a wire
list, it returns a flat list of gates to apply. It does not touch PennyLane's
``Operation``/``QNode``/tape/device machinery at all. That means the same
logic can be re-targeted at Qiskit's ``QuantumCircuit`` instead of PennyLane's
own operator representation, and the result is a real, standalone-executable
circuit -- unlike ``upstream/qaoa`` etc., whose modules import
PennyLane's tape/measurement machinery and cannot run standalone.
"""
from algorithms_extended.pennylane_asymmetrica.templates.layers import (
    basic_entangler_layers,
    basic_entangler_shape,
    strongly_entangling_layers,
    strongly_entangling_shape,
)
from algorithms_extended.pennylane_asymmetrica.templates.embeddings import (
    angle_embedding,
    qaoa_embedding,
    qaoa_embedding_shape,
)

__all__ = [
    "basic_entangler_layers",
    "basic_entangler_shape",
    "strongly_entangling_layers",
    "strongly_entangling_shape",
    "angle_embedding",
    "qaoa_embedding",
    "qaoa_embedding_shape",
]
