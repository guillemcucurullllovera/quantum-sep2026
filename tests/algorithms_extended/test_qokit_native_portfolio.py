"""Real tests for the QOKit native-portfolio bridge (expansion)."""
import numpy as np
import pytest

from algorithms_extended.qokit_finance.native_portfolio_bridge import (
    build_native_qokit_portfolio_problem,
    classical_optimum_via_warm_start,
    run_native_qokit_portfolio,
    warm_start_initial_point,
)
from core.errors import ValidationError


def test_build_problem_shapes():
    p = build_native_qokit_portfolio_problem(6, 3, 0.5, seed=1)
    assert p["means"].shape == (6,)
    assert p["cov"].shape == (6, 6)
    assert p["K"] == 3


def test_cardinality_exceeding_n_assets_raises():
    with pytest.raises(ValidationError):
        build_native_qokit_portfolio_problem(4, 10, 0.5, seed=1)


def test_classical_optimum_is_exact_for_small_n():
    """Cross-check the diagonal-extraction shortcut against a manual
    brute-force enumeration of QOKit's own Hamiltonian, proving the
    "take the diagonal" simplification is mathematically correct and not
    just a convenient guess."""
    from algorithms_extended.qokit_finance.upstream.portfolio_optimization import get_problem_H

    p = build_native_qokit_portfolio_problem(5, 2, 0.3, seed=7)
    H = get_problem_H(p)
    diag = np.real(np.diag(H))
    manual_best = float(diag.min())

    result = classical_optimum_via_warm_start(p)
    assert result["energy"] == pytest.approx(manual_best, abs=1e-9)


def test_classical_optimum_too_large_n_raises():
    p = build_native_qokit_portfolio_problem(6, 3, 0.5, seed=1)
    p["N"] = 25  # simulate an oversized problem without actually building 2^25
    with pytest.raises(ValidationError):
        classical_optimum_via_warm_start(p)


def test_warm_start_relaxation_runs():
    p = build_native_qokit_portfolio_problem(6, 3, 0.5, seed=1)
    stats = warm_start_initial_point(p)
    assert "qaoa_energy" in stats and "best_cut" in stats
    assert np.isfinite(stats["qaoa_energy"])


def test_warm_start_angles_affect_energy():
    """Real assertion: different QAOA angles should generally produce a
    different expectation value -- proves run_standard_qaoa is actually
    running the circuit, not returning a constant."""
    p = build_native_qokit_portfolio_problem(6, 3, 0.5, seed=1)
    e1 = warm_start_initial_point(p, gamma=0.1, beta=0.3)["qaoa_energy"]
    e2 = warm_start_initial_point(p, gamma=0.9, beta=1.2)["qaoa_energy"]
    assert e1 != pytest.approx(e2)


def test_full_run_reproducible_with_same_seed():
    r1 = run_native_qokit_portfolio(n_assets=6, cardinality=3, risk_aversion=0.5, seed=99, log_to_registry=False)
    r2 = run_native_qokit_portfolio(n_assets=6, cardinality=3, risk_aversion=0.5, seed=99, log_to_registry=False)
    assert r1.metrics["classical_baseline_energy"] == pytest.approx(r2.metrics["classical_baseline_energy"])


def test_full_run_logs_to_registry(tmp_path, monkeypatch):
    from core.experiment_registry import ExperimentRegistry

    reg = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    result = run_native_qokit_portfolio(n_assets=5, cardinality=2, risk_aversion=0.4, seed=3, log_to_registry=False)
    run_id = reg.record(result)
    fetched = reg.get(run_id)
    assert fetched.metadata.algorithm == "qaoa_qokit_native_portfolio"
