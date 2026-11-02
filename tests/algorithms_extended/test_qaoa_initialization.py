import numpy as np
import pytest

from algorithms_extended.qaoa_initialization import (
    INIT_STRATEGY_REGISTRY,
    interpolate_to_next_depth,
    linear_ramp_initialization,
    random_initialization,
)
from core.errors import ValidationError


def test_interp_produces_one_more_layer():
    gammas = np.array([0.5, 1.0, 1.5])
    betas = np.array([1.0, 0.7, 0.4])
    new_g, new_b = interpolate_to_next_depth(gammas, betas)
    assert len(new_g) == len(gammas) + 1
    assert len(new_b) == len(betas) + 1


def test_interp_boundary_matches_formula_for_p1():
    # p=1: single-layer input. New layer 1 uses only gamma_0=0 weighted, layer 2 only gamma_1
    gammas = np.array([2.0])
    betas = np.array([1.0])
    new_g, new_b = interpolate_to_next_depth(gammas, betas)
    # i=1: w_prev=0, w_next=1 -> uses padded_gamma[1] = original gamma[0]
    assert new_g[0] == pytest.approx(2.0)
    # i=2: w_prev=1, w_next=0 -> uses padded_gamma[1] = original gamma[0]
    assert new_g[1] == pytest.approx(2.0)


def test_interp_rejects_mismatched_shapes():
    with pytest.raises(ValidationError):
        interpolate_to_next_depth(np.array([1.0, 2.0]), np.array([1.0]))


def test_interp_is_deterministic_unlike_random():
    g1, b1 = interpolate_to_next_depth(np.array([1.0, 2.0]), np.array([0.5, 0.3]))
    g2, b2 = interpolate_to_next_depth(np.array([1.0, 2.0]), np.array([0.5, 0.3]))
    assert np.array_equal(g1, g2) and np.array_equal(b1, b2)

    r1, _ = random_initialization(p=2, seed=1)
    r2, _ = random_initialization(p=2, seed=2)
    assert not np.array_equal(r1, r2)  # different seeds -> different draws, proves it's actually random


def test_linear_ramp_shape_and_monotonicity():
    gammas, betas = linear_ramp_initialization(p=5)
    assert len(gammas) == 5 and len(betas) == 5
    assert np.all(np.diff(gammas) > 0)  # gamma strictly increasing
    assert np.all(np.diff(betas) < 0)  # beta strictly decreasing


def test_linear_ramp_respects_bounds():
    gamma_max, beta_max = 1.0, 0.5
    gammas, betas = linear_ramp_initialization(p=4, gamma_max=gamma_max, beta_max=beta_max)
    assert gammas[-1] == pytest.approx(gamma_max)
    assert betas[0] < beta_max  # first layer is beta_max * (1 - 1/p), strictly less than beta_max


def test_random_initialization_rejects_invalid_p():
    with pytest.raises(ValidationError):
        random_initialization(p=0)


def test_registry_exposes_all_three_strategies():
    assert set(INIT_STRATEGY_REGISTRY.list()) == {"linear_ramp", "random", "interp_from_previous"}


def test_registry_linear_ramp_matches_direct_call():
    fn = INIT_STRATEGY_REGISTRY.get("linear_ramp")
    g1, b1 = fn(p=3)
    g2, b2 = linear_ramp_initialization(p=3)
    assert np.array_equal(g1, g2) and np.array_equal(b1, b2)
