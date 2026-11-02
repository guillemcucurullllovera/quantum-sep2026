"""QuadraticProgram-style finance modeling layer, materially adapted from
qiskit-optimization's object model (named variables, explicit constraint
objects, a converter pipeline) but built as a front end onto this
platform's own, already-correct QUBO builder rather than a parallel
reimplementation of qiskit-optimization's `QuadraticProgram`/`to_ising`
machinery. See THIRD_PARTY_NOTICES.md for exactly what was adapted vs. written
fresh, and LICENSES/NOTICE entry for qiskit-optimization (Apache-2.0).
"""
from algorithms_extended.qiskit_optimization_finance.model import FinancePortfolioModel, SectorRange
from algorithms_extended.qiskit_optimization_finance.converters import (
    AdaptivePenaltyStrategy,
    FixedPenaltyStrategy,
    ModelToQUBOConverter,
)
from algorithms_extended.qiskit_optimization_finance.repair import is_feasible, repair_infeasible
from algorithms_extended.qiskit_optimization_finance.runner import solve_model

__all__ = [
    "FinancePortfolioModel",
    "SectorRange",
    "AdaptivePenaltyStrategy",
    "FixedPenaltyStrategy",
    "ModelToQUBOConverter",
    "is_feasible",
    "repair_infeasible",
    "solve_model",
]
