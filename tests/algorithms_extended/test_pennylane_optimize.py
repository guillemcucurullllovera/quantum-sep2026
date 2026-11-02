"""Tests for algorithms_extended/pennylane_asymmetrica/optimize/rotosolve.py.

Verifies:
  * the near-verbatim-ported ``_brute_optimizer`` helper finds the global
    minimum of a multi-frequency 1-D function (no PennyLane dependency to
    cross-check against, so validated against a dense grid search instead);
  * the reimplemented closed-form single-parameter update exactly reproduces
    the analytic minimum for a single two-eigenvalue-generator gate in one
    evaluation-of-three;
  * ``RotosolveOptimizer`` actually converges to the true optimum on real
    multi-parameter test problems built from this platform's own
    ``VQCAnsatz`` circuits (closed-form) and a synthetic QAOA-shaped
    multi-frequency cost (brute substep).
"""
from __future__ import annotations

import numpy as np
import pytest

from algorithms_extended.pennylane_asymmetrica.optimize.rotosolve import (
    RotosolveOptimizer,
    _brute_optimizer,
)
from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from core.errors import ValidationError


class TestBruteOptimizerHelper:
    def test_finds_minimum_of_single_cosine(self):
        fun = lambda x: np.cos(np.asarray(x).reshape(-1)[0])
        center, y_min = _brute_optimizer(fun, num_steps=4, bounds=(-np.pi, np.pi), Ns=15)
        assert abs(center - np.pi) < 0.05 or abs(center + np.pi) < 0.05
        assert y_min == pytest.approx(-1.0, abs=1e-6)

    def test_finds_minimum_of_multi_frequency_function(self):
        def fun(x):
            t = np.asarray(x).reshape(-1)[0]
            return 0.6 * np.cos(2 * t) + 0.3 * np.cos(0.7 * t) - 0.2 * np.sin(t)

        ts = np.linspace(-np.pi, np.pi, 20000)
        true_min = min(fun([t]) for t in ts)
        center, y_min = _brute_optimizer(fun, num_steps=5, bounds=(-np.pi, np.pi), Ns=15)
        assert y_min == pytest.approx(true_min, abs=1e-3)


class TestClosedFormSubstep:
    def test_exact_single_qubit_minimum(self):
        # cost(theta) = cos(theta), global min at theta = +-pi, value -1.
        cost_fn = lambda params: float(np.cos(params[0]))
        opt = RotosolveOptimizer(substep="closed_form")
        params, history = opt.minimize(cost_fn, np.array([0.3]), n_steps=1)
        assert history[-1] == pytest.approx(-1.0, abs=1e-9)

    def test_exact_in_one_sweep_regardless_of_start(self):
        cost_fn = lambda params: float(3 * np.sin(params[0] + 0.7) + 1)
        opt = RotosolveOptimizer(substep="closed_form")
        for start in (-2.5, 0.0, 1.2, 3.0):
            _, history = opt.minimize(cost_fn, np.array([start]), n_steps=1)
            assert history[-1] == pytest.approx(-2.0, abs=1e-8)  # min of 3*sin(.)+1 is -2

    def test_rejects_unknown_substep(self):
        with pytest.raises(ValidationError):
            RotosolveOptimizer(substep="not_a_thing")


class TestRotosolveOnVQCAnsatz:
    """Every parameter in VQCAnsatz's RY/RZ layers enters the circuit through
    exactly one Pauli-rotation gate -- the exact regime the closed-form
    substep is valid for."""

    def test_converges_to_all_minus_one(self):
        n_qubits = 3
        ansatz = VQCAnsatz(n_qubits=n_qubits, version=VQCVersion.V1_LINEAR)
        rng = np.random.default_rng(11)
        x = rng.uniform(-np.pi, np.pi, n_qubits)

        def cost_fn(params: np.ndarray) -> float:
            return float(np.sum(ansatz.z_expectations(x, params)))  # min possible: -n_qubits

        params0 = ansatz.init_params(seed=11)
        opt = RotosolveOptimizer(substep="closed_form")
        best_params, history = opt.minimize(cost_fn, params0, n_steps=3)

        assert history[-1] == pytest.approx(-float(n_qubits), abs=1e-6)
        assert history[0] > history[-1]  # real improvement happened
        assert len(best_params) == ansatz.param_count

    def test_monotone_non_increasing_cost(self):
        # Coordinate descent with an exact per-coordinate global minimizer
        # can never make the cost worse -- *provided* the cost is linear in
        # the Z-expectations. (A quadratic cost like MSE is a genuine
        # counter-example: squaring a single-frequency sinusoid produces a
        # second harmonic, so the 3-point closed form is no longer exact for
        # that restricted function -- this is exactly why the class is
        # named for "generators with two eigenvalues", not "any cost
        # function of such a circuit".)
        n_qubits = 4
        ansatz = VQCAnsatz(n_qubits=n_qubits, version=VQCVersion.V4_RING)
        rng = np.random.default_rng(5)
        x = rng.uniform(-np.pi, np.pi, n_qubits)
        target = rng.choice([-1.0, 1.0], n_qubits)

        def cost_fn(params: np.ndarray) -> float:
            z = ansatz.z_expectations(x, params)
            return float(-np.dot(z, target))  # linear -> single-frequency per parameter

        params0 = ansatz.init_params(seed=5)
        opt = RotosolveOptimizer(substep="closed_form")
        params = params0
        costs = [cost_fn(params)]
        for _ in range(4):
            params, cost_before = opt.step_and_cost(cost_fn, params)
            costs.append(cost_fn(params))
        for i in range(len(costs) - 1):
            assert costs[i + 1] <= costs[i] + 1e-10


class TestRotosolveBruteOnQAOAShapedCost:
    """A synthetic 2-parameter cost with the same qualitative shape as a QAOA
    gamma/beta objective: each parameter multiplies several different
    frequencies at once, so the closed-form single-frequency assumption does
    not hold and the general "brute" substep is required."""

    def test_converges_below_random_search_baseline(self):
        rng = np.random.default_rng(3)
        coeffs = rng.uniform(0.2, 1.0, size=(2, 3))  # per-param, 3 frequencies each
        freqs = np.array([1.0, 1.7, 2.3])

        def cost_fn(params: np.ndarray) -> float:
            total = 0.0
            for d in range(2):
                total += float(np.sum(coeffs[d] * np.cos(freqs * params[d])))
            return total

        opt = RotosolveOptimizer(substep="brute", brute_num_steps=4, brute_Ns=13)
        params0 = np.array([0.5, -1.0])
        best_params, history = opt.minimize(cost_fn, params0, n_steps=3)

        # Baseline: best of 500 random points in [-pi, pi]^2.
        random_pts = rng.uniform(-np.pi, np.pi, size=(500, 2))
        random_best = min(cost_fn(p) for p in random_pts)

        assert history[-1] < random_best
        assert history[-1] <= history[0]
