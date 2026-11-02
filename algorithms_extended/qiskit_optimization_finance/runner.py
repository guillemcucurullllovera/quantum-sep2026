"""End-to-end: FinancePortfolioModel -> QUBO -> QAOA solve -> feasibility
report (+ repair if needed). This is the actual "QAOA integration" this
module exists to provide -- the modeling/converter/repair pieces above are
infrastructure, this is where they get used together.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms.qaoa_qubo.qubo_builder import LongShortConstraints, PortfolioConstraints
from algorithms_extended.qiskit_optimization_finance.converters import ModelToQUBOConverter, PenaltyStrategy
from algorithms_extended.qiskit_optimization_finance.model import FinancePortfolioModel
from algorithms_extended.qiskit_optimization_finance.repair import is_feasible, repair_infeasible


@dataclass
class ModelSolveResult:
    bitstring: np.ndarray
    cost: float
    feasibility: dict[str, bool]
    was_repaired: bool
    engine: str


def solve_model(
    model: FinancePortfolioModel,
    strategy: PenaltyStrategy,
    solver,
    *,
    repair: bool = True,
    seed: int = 0,
) -> ModelSolveResult:
    """`solver` is any object exposing `.solve(Q) -> result` with a
    `.best_bitstring` attribute (a 0/1 string) -- both
    `algorithms.qaoa_qubo.solvers.QiskitQAOASolver` and `.NumpyQAOASolver`
    satisfy this, so this function works with either without caring which.
    """
    Q, constraints = ModelToQUBOConverter().convert(model, strategy)
    result = solver.solve(Q)
    x = np.array([int(b) for b in result.best_bitstring], dtype=float)

    was_repaired = False
    if model.long_short:
        # LongShortConstraints has no is_feasible() check implemented here yet
        # (mutual-exclusion + leverage are penalty-encoded only) -- report
        # what we can, don't silently claim full feasibility.
        feasibility = {"note_long_short_repair_not_implemented": True}
    else:
        assert isinstance(constraints, PortfolioConstraints)
        feasibility = is_feasible(x, constraints)
        if repair and not all(feasibility.values()):
            x, feasibility = repair_infeasible(Q, x, constraints, seed=seed)
            was_repaired = True

    cost = float(x @ Q @ x)
    return ModelSolveResult(
        bitstring=x, cost=cost, feasibility=feasibility,
        was_repaired=was_repaired, engine=getattr(result, "engine", type(solver).__name__),
    )
