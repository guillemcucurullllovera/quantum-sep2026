import numpy as np

from experiments.search import GaussianProcess, bayesian_search


def test_gp_predicts_training_points_near_exactly():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 4.0])
    gp = GaussianProcess(length_scale=1.0, noise=1e-8).fit(X, y)
    mean, var = gp.predict(X)
    assert np.allclose(mean, y, atol=1e-3)
    assert np.all(var < 1e-3)  # near-zero uncertainty exactly at training points


def test_bayesian_search_beats_random_on_a_simple_quadratic():
    def objective(x):
        return float((x[0] - 3.0) ** 2)  # minimum at x=3, value 0

    result = bayesian_search(objective, bounds=[(-10, 10)], n_initial=5, n_iterations=15, seed=0)
    rng = np.random.default_rng(0)
    random_values = [objective(rng.uniform(-10, 10, size=1)) for _ in range(20)]
    assert result.best_value <= min(random_values)  # BO should find at least as good a point with fewer evals
    assert abs(result.best_params[0] - 3.0) < 1.0
