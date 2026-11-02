import numpy as np

from classical.optimization import brute_force_qubo, simulated_annealing_qubo, milp_binary_qp


def _small_qubo(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(6, 6))
    Q = (A + A.T) / 2  # symmetric, indefinite in general
    return Q


def test_brute_force_matches_manual_check():
    # diag(-1,-1,1) => minimizing x^T Q x picks x=(1,1,0), obj=-2
    Q = np.diag([-1.0, -1.0, 1.0])
    x, obj = brute_force_qubo(Q)
    assert list(x) == [1.0, 1.0, 0.0]
    assert obj == -2.0


def test_brute_force_rejects_too_large_instance():
    Q = np.zeros((25, 25))
    try:
        brute_force_qubo(Q, max_vars=24)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_simulated_annealing_reaches_brute_force_optimum_on_small_instance():
    Q = _small_qubo()
    _, exact_obj = brute_force_qubo(Q)
    _, sa_obj = simulated_annealing_qubo(Q, n_iters=8000, seed=1)
    assert abs(sa_obj - exact_obj) < 1e-9  # small enough instance that SA should find the exact optimum


def test_milp_respects_budget_constraint():
    # minimize -c^T x (i.e. maximize c^T x) s.t. sum(x) <= 2, x binary
    c = np.array([-3.0, -1.0, -2.0, -0.5])
    x, obj = milp_binary_qp(c, A_ub=np.ones((1, 4)), b_ub=np.array([2.0]))
    assert x.sum() <= 2.0 + 1e-9
    # optimal picks the two largest-magnitude coefficients: indices 0 and 2
    assert x[0] == 1.0 and x[2] == 1.0
    assert abs(obj - (-5.0)) < 1e-6
