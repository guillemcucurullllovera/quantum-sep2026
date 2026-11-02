"""QAOA / QUBO portfolio optimization algorithm family.

See README.md in this directory for the Gen1 vs Gen2 history and why both
are kept. `qubo_builder.py`, `solvers.py`, and `classical_baseline.py` are
the net-new unified layer built on top of both generations for this
platform (constraint suite, Qiskit hardware-aware execution path, and the
classical comparators every quantum result is checked against).
"""
from core.registry import ALGORITHM_REGISTRY
from algorithms.qaoa_qubo.solvers import solve_qubo

ALGORITHM_REGISTRY.register("qaoa_qubo")(solve_qubo)
