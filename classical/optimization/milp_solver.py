"""MILP wrapper (scipy.optimize.milp) for linearly-constrained binary QPs.

QUBO objectives are quadratic, which milp cannot handle directly, so this
module targets the *linear* portfolio/allocation subproblems (e.g. cardinality
and budget-constrained linear allocation) rather than pretending to solve
general QUBOs exactly at scale. Use brute_force_qubo/simulated_annealing_qubo
for the quadratic case.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def milp_binary_qp(
    c: np.ndarray,
    *,
    A_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Minimize c^T x over binary x subject to A_ub x <= b_ub and A_eq x == b_eq.

    A real linear baseline for e.g. cardinality/budget-constrained asset
    selection, complementing the quadratic QUBO solvers above.
    """
    c = np.asarray(c, dtype=float)
    n = len(c)
    integrality = np.ones(n)
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

    constraints = []
    if A_ub is not None:
        constraints.append(LinearConstraint(np.asarray(A_ub), -np.inf, np.asarray(b_ub)))
    if A_eq is not None:
        b_eq_arr = np.asarray(b_eq)
        constraints.append(LinearConstraint(np.asarray(A_eq), b_eq_arr, b_eq_arr))

    result = milp(c=c, constraints=constraints or None, bounds=bounds, integrality=integrality)
    if not result.success:
        raise RuntimeError(f"MILP solve failed: {result.message}")
    return np.round(result.x), float(result.fun)
