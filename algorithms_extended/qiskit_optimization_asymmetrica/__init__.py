# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# This file has been altered from the original qiskit-optimization __init__.py:
# it is a package-init adapted to this platform's namespace
# (algorithms_extended.qiskit_optimization_asymmetrica instead of qiskit_optimization).
# The original upstream __init__.py is preserved verbatim, unaltered, alongside this
# file as _upstream_init_reference.py for provenance comparison. See THIRD_PARTY_NOTICES.md.
"""Asymmetrica full-base adaptation of qiskit-optimization (Workstream 2).

See THIRD_PARTY_NOTICES.md for what was brought in verbatim vs. materially adapted.
"""

from .exceptions import AlgorithmError, QiskitOptimizationError
from .infinity import INFINITY  # must be at the top of the file
from .problems.quadratic_program import QuadraticProgram
from ._upstream_version import __version__

__all__ = [
    "__version__",
    "QuadraticProgram",
    "QiskitOptimizationError",
    "AlgorithmError",
    "INFINITY",
]
