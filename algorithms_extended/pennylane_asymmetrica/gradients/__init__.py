"""Parameter-shift-rule gradients, adapted from PennyLane's
``pennylane.gradients`` (Apache-2.0, Xanadu Quantum Technologies Inc.).

See ``THIRD_PARTY_NOTICES.md`` (section): ``pennylane/gradients/
parameter_shift.py`` and ``general_shift_rules.py`` are not portable as
files (they import ``pennylane.tape.QuantumScript``, ``pennylane.operation
.Operator``, ``pennylane.measurements``, and the ``transforms.core`` tape-
transform framework throughout). What *is* separable is the underlying
two-term shift-rule mathematics for a generator with eigenvalues
:math:`\\pm 1/2` (any Pauli rotation gate), which this module reimplements
directly against a caller-supplied Qiskit-circuit-evaluation callable.
"""
from algorithms_extended.pennylane_asymmetrica.gradients.parameter_shift import (
    parameter_shift_gradient,
    reupload_ansatz_z_expectations,
)

__all__ = ["parameter_shift_gradient", "reupload_ansatz_z_expectations"]
