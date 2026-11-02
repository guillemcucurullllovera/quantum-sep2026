"""Rotosolve: a gradient-free coordinate-descent optimizer for variational
quantum circuits, adapted from PennyLane's ``pennylane/optimize/rotosolve.py``
(Apache-2.0, Xanadu Quantum Technologies Inc.).

Two pieces were adapted, with two different provenance statuses:

1. ``_brute_optimizer`` below is a **near-verbatim port** of upstream's
   private helper of the same name. It is pure ``scipy.optimize.brute`` +
   arithmetic -- the one piece of ``rotosolve.py`` that never imports
   ``pennylane`` at all -- so it needed no rewrite, only a docstring and a
   `sp` -> `scipy` rename for clarity. It performs an iterative
   (coarse-to-fine) 1-D global search over a bounded interval and makes
   Rotosolve usable on a cost function with an *unknown* frequency spectrum
   in a single parameter (which is exactly ``QiskitQAOASolver``'s situation:
   a QAOA ``gamma``/``beta`` angle multiplies several different Ising
   coefficients at once, so the cost is a sum of several different
   frequencies in that one angle -- not the single-frequency case below).

2. The ``"closed_form"`` substep is a **from-scratch reimplementation** of
   the closed-form update rule documented in upstream's ``RotosolveOptimizer``
   docstring (itself citing Ostaszewski, Grant & Benedetti 2021,
   `Quantum 5, 391 <https://quantum-journal.org/papers/q-2021-01-28-391/>`_):
   for a parameter that enters the circuit through a *single* gate whose
   generator has exactly two eigenvalues (i.e. any Pauli rotation --
   RX/RY/RZ), the restriction of the cost function to that one parameter is
   an exact single-frequency sinusoid, and its global minimum has a closed
   form computable from just three cost evaluations:

   .. math::
        \\theta_d^* = -\\frac{\\pi}{2} - \\mathrm{arctan2}\\!\\left(
            2 H(0) - H(\\pi/2) - H(-\\pi/2),\\; H(\\pi/2) - H(-\\pi/2)\\right)

   where :math:`H(\\theta_d)` is the cost function with every parameter
   except :math:`\\theta_d` held fixed. This code is new -- upstream's own
   implementation of this branch is entangled with ``pennylane.workflow.QNode``
   dispatch and was not usable as a starting point -- but the *mathematics*
   is faithfully reproduced from the cited formula, not guessed.

   Not ported: upstream's general multi-frequency case, which reconstructs
   the restricted cost function via ``pennylane.fourier.reconstruct`` (a
   trigonometric-interpolation module that itself depends on
   ``pennylane.workflow.QNode`` for automatic frequency-spectrum
   introspection from the circuit's gate generators). The ``"brute"``
   substep here is a general-purpose (if less sample-efficient) substitute
   for that case: it makes no assumption about the frequency spectrum at
   all, at the cost of needing ``brute_num_steps * brute_Ns`` evaluations
   per parameter per step instead of 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import scipy.optimize as scipy_optimize

from core.errors import ValidationError

CostFn = Callable[[np.ndarray], float]


def _brute_optimizer(
    fun: Callable[[float], float],
    num_steps: int,
    bounds: tuple[float, float],
    Ns: int,
) -> tuple[float, float]:
    """Near-verbatim port of ``pennylane.optimize.rotosolve._brute_optimizer``:
    iteratively refine a 1-D ``scipy.optimize.brute`` grid search, evaluating
    ``fun`` at ``Ns`` equally spaced points and narrowing the search window
    around the best point ``num_steps`` times.
    """
    width = bounds[1] - bounds[0]
    center = (bounds[1] + bounds[0]) / 2
    y_min = fun(center)
    for _ in range(num_steps):
        window = (center - width / 2, center + width / 2)
        result = scipy_optimize.brute(fun, ranges=(window,), full_output=True, Ns=Ns)
        center_arr, y_min = result[0], result[1]
        center = float(center_arr[0])
        width /= Ns
    return center, float(y_min)


@dataclass
class RotosolveOptimizer:
    """Coordinate-descent optimizer: each step sweeps every parameter once,
    replacing it with the exact (or, for ``substep="brute"``, approximate)
    global minimizer of the cost function restricted to that one coordinate,
    with all others held fixed.

    Args:
        substep: ``"closed_form"`` (default) uses the exact 3-evaluation
            formula and is only valid when the *cost function itself* -- not
            just the circuit -- is a single-frequency sinusoid in each
            parameter: every parameter must enter the circuit via a single
            two-eigenvalue-generator gate (true of e.g.
            ``algorithms.vqc_qml.ansatz.VQCAnsatz``'s per-qubit RY/RZ
            layers), *and* the cost must be linear in the resulting
            expectation value(s) -- a raw expectation or a weighted sum of
            them, not e.g. a squared error, since squaring a single-
            frequency sinusoid introduces a second harmonic. Both
            conditions are exercised directly in
            ``tests/algorithms_extended/test_pennylane_optimize.py``.
            ``"brute"`` makes no such assumption (true of QAOA's
            ``gamma``/``beta`` angles, each of which multiplies several
            different Ising coefficients within one layer) at the cost of
            more evaluations.
        brute_num_steps, brute_Ns, bounds: only used for ``substep="brute"``.
    """

    substep: Literal["closed_form", "brute"] = "closed_form"
    brute_num_steps: int = 4
    brute_Ns: int = 15
    bounds: tuple[float, float] = (-np.pi, np.pi)

    def __post_init__(self) -> None:
        if self.substep not in ("closed_form", "brute"):
            raise ValidationError(f"RotosolveOptimizer: unknown substep '{self.substep}'")

    @staticmethod
    def _closed_form_argmin(restricted: Callable[[float], float]) -> float:
        h0 = restricted(0.0)
        h_plus = restricted(np.pi / 2)
        h_minus = restricted(-np.pi / 2)
        theta_star = -np.pi / 2 - np.arctan2(2 * h0 - h_plus - h_minus, h_plus - h_minus)
        # wrap into (-pi, pi]
        return float((theta_star + np.pi) % (2 * np.pi) - np.pi)

    def step_and_cost(self, cost_fn: CostFn, params: np.ndarray) -> tuple[np.ndarray, float]:
        """One full sweep over all parameters. Returns (new_params, cost_before_this_step)."""
        params = np.array(params, dtype=float)
        cost_before = float(cost_fn(params))
        for d in range(len(params)):
            def restricted(theta_d: float, _d: int = d) -> float:
                # scipy.optimize.brute calls this with a length-1 array, not a
                # bare float -- normalize either way.
                p = params.copy()
                p[_d] = float(np.asarray(theta_d).reshape(-1)[0])
                return float(cost_fn(p))

            if self.substep == "closed_form":
                theta_star = self._closed_form_argmin(restricted)
            else:
                theta_star, _ = _brute_optimizer(
                    restricted, self.brute_num_steps, self.bounds, self.brute_Ns
                )
            params[d] = theta_star
        return params, cost_before

    def step(self, cost_fn: CostFn, params: np.ndarray) -> np.ndarray:
        new_params, _ = self.step_and_cost(cost_fn, params)
        return new_params

    def minimize(
        self, cost_fn: CostFn, params0: np.ndarray, n_steps: int = 30
    ) -> tuple[np.ndarray, list[float]]:
        """Run ``n_steps`` full sweeps. Returns (best_params, cost_history) where
        ``cost_history`` has ``n_steps + 1`` entries (before each step, plus the
        final cost)."""
        params = np.array(params0, dtype=float)
        history: list[float] = []
        for _ in range(n_steps):
            params, cost_before = self.step_and_cost(cost_fn, params)
            history.append(cost_before)
        history.append(float(cost_fn(params)))
        return params, history
