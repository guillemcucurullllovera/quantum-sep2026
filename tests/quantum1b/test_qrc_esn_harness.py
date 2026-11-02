"""Correctness tests for the Track C (QRC vs ESN) harness: synthetic-task
determinism, baseline sanity, ESN/QRC numerical health, and the
real-financial-task BLOCKED guard. Mirrors `test_foundation.py`'s style.

Kept deliberately small-scale (short series, small readout dims, few
seeds) so the suite runs fast -- the full preregistered EXP-C1 grid is
exercised by `experiments/quantum1b/run_track_c.py`, not by this suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.quantum1b.qrc_esn_harness import (
    RealFinancialExecutionBlocked,
    compare_qrc_vs_esn,
    memory_capacity,
    paired_significance,
    qlike_loss,
    run_esn,
    run_persistence_baseline,
    run_qrc,
    run_real_financial_task,
    run_ridge_raw_baseline,
)
from experiments.quantum1b.synthetic_vol_task import generate_synthetic_realized_vol_proxy


class TestSyntheticTaskDeterminism:
    def test_same_seed_same_series(self):
        a = generate_synthetic_realized_vol_proxy(n=500, seed=3)
        b = generate_synthetic_realized_vol_proxy(n=500, seed=3)
        np.testing.assert_array_equal(a.proxy, b.proxy)
        np.testing.assert_array_equal(a.sigma2, b.sigma2)

    def test_different_seed_different_series(self):
        a = generate_synthetic_realized_vol_proxy(n=500, seed=0)
        b = generate_synthetic_realized_vol_proxy(n=500, seed=1)
        assert not np.array_equal(a.proxy, b.proxy)

    def test_series_always_positive(self):
        sv = generate_synthetic_realized_vol_proxy(n=1000, seed=7)
        assert np.all(sv.proxy > 0)

    def test_series_length_matches_n(self):
        sv = generate_synthetic_realized_vol_proxy(n=321, seed=0)
        assert len(sv.proxy) == 321
        assert len(sv.sigma2) == 321

    def test_rejects_non_stationary_params(self):
        with pytest.raises(ValueError):
            generate_synthetic_realized_vol_proxy(n=100, seed=0, alpha=0.6, beta=0.6)

    def test_params_recorded(self):
        sv = generate_synthetic_realized_vol_proxy(n=100, seed=0)
        assert sv.params["process"] == "synthetic_garch11_realized_vol_proxy"
        assert sv.seed == 0


class TestQlikeAndMemoryCapacity:
    def test_qlike_zero_at_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 0.5])
        assert qlike_loss(actual, actual) == pytest.approx(0.0, abs=1e-9)

    def test_qlike_none_when_no_positive_predictions(self):
        actual = np.array([1.0, 2.0])
        pred = np.array([-1.0, 0.0])
        assert qlike_loss(actual, pred) is None

    def test_qlike_nonnegative(self):
        rng = np.random.default_rng(0)
        actual = rng.uniform(0.1, 2.0, size=50)
        pred = rng.uniform(0.1, 2.0, size=50)
        val = qlike_loss(actual, pred)
        assert val is not None
        assert val >= -1e-9

    def test_memory_capacity_bounded_per_delay(self):
        rng = np.random.default_rng(0)
        u = rng.uniform(-1, 1, size=200)
        states = rng.normal(size=(200, 4))
        mc = memory_capacity(states, u, max_delay=5)
        assert mc["total"] >= 0.0
        for d in mc["per_delay"]:
            assert 0.0 <= d["capacity"] <= 1.0

    def test_memory_capacity_high_for_identity_readout(self):
        # states[t] == [u[t], u[t-1]] should give near-perfect delay-0 and
        # delay-1 reconstruction -- a sanity check that the diagnostic
        # itself is wired correctly (not that any particular reservoir is).
        rng = np.random.default_rng(0)
        u = rng.uniform(-1, 1, size=300)
        states = np.zeros((300, 2))
        states[:, 0] = u
        states[1:, 1] = u[:-1]
        mc = memory_capacity(states, u, max_delay=1, washout=5)
        assert mc["per_delay"][0]["capacity"] > 0.9
        assert mc["per_delay"][1]["capacity"] > 0.9


class TestBaselinesSane:
    def test_ridge_raw_beats_persistence_on_structured_series(self):
        sv = generate_synthetic_realized_vol_proxy(n=1200, seed=0)
        series = sv.proxy
        L = len(series) - 1
        split = int(0.7 * L)
        r_persist = run_persistence_baseline(series, split, seed=0)
        r_ridge = run_ridge_raw_baseline(series, split, seed=0)
        assert np.isfinite(r_persist.oos_mse)
        assert np.isfinite(r_ridge.oos_mse)
        assert r_ridge.oos_mse < r_persist.oos_mse

    def test_persistence_readout_dim_zero(self):
        sv = generate_synthetic_realized_vol_proxy(n=300, seed=0)
        series = sv.proxy
        L = len(series) - 1
        r = run_persistence_baseline(series, int(0.7 * L), seed=0)
        assert r.readout_dim == 0
        assert r.method == "persistence"


class TestEsnAndQrcNumericalHealth:
    @staticmethod
    @pytest.fixture(scope="class")
    def small_series():
        sv = generate_synthetic_realized_vol_proxy(n=500, seed=0)
        return sv.proxy

    def test_esn_forecast_finite(self, small_series):
        series = small_series
        L = len(series) - 1
        split = int(0.7 * L)
        r = run_esn(series, split, readout_dim=4, seed=0)
        assert np.isfinite(r.oos_mse)
        assert r.oos_mse >= 0.0
        assert r.qlike is None or np.isfinite(r.qlike)
        assert np.isfinite(r.extra["memory_capacity"]["total"])
        assert r.method == "esn"
        assert r.readout_dim == 4

    def test_qrc_forecast_finite(self, small_series):
        series = small_series
        L = len(series) - 1
        split = int(0.7 * L)
        r = run_qrc(series, split, readout_dim=4, seed=0)
        assert np.isfinite(r.oos_mse)
        assert r.oos_mse >= 0.0
        assert r.qlike is None or np.isfinite(r.qlike)
        assert np.isfinite(r.extra["memory_capacity"]["total"])
        assert r.method == "qrc"
        assert r.readout_dim == 4

    def test_qrc_readout_dim_matches_n1_plus_n2(self, small_series):
        series = small_series
        L = len(series) - 1
        split = int(0.7 * L)
        r = run_qrc(series, split, readout_dim=6, seed=1)
        assert r.extra["n1"] + r.extra["n2"] == 6

    def test_esn_and_qrc_deterministic_given_seed(self, small_series):
        series = small_series
        L = len(series) - 1
        split = int(0.7 * L)
        r1 = run_esn(series, split, readout_dim=4, seed=2)
        r2 = run_esn(series, split, readout_dim=4, seed=2)
        assert r1.oos_mse == pytest.approx(r2.oos_mse)
        q1 = run_qrc(series, split, readout_dim=4, seed=2)
        q2 = run_qrc(series, split, readout_dim=4, seed=2)
        assert q1.oos_mse == pytest.approx(q2.oos_mse)


class TestCompareQrcVsEsn:
    def test_reduced_grid_structure(self):
        result = compare_qrc_vs_esn(readout_dims=(4,), seeds=(0, 1), n_points=500)
        assert set(result["esn"].keys()) == {4}
        assert set(result["qrc"].keys()) == {4}
        assert len(result["esn"][4]) == 2
        assert len(result["qrc"][4]) == 2
        assert len(result["persistence"]) == 2
        assert len(result["ridge_raw"]) == 2
        assert len(result["paired_esn_minus_qrc_mse"][4]) == 2
        for r in result["esn"][4] + result["qrc"][4]:
            assert np.isfinite(r.oos_mse)

    def test_paired_significance_shapes(self):
        result = compare_qrc_vs_esn(readout_dims=(4,), seeds=(0, 1, 2), n_points=500)
        sig = paired_significance(result["paired_esn_minus_qrc_mse"][4])
        assert sig["n"] == 3
        assert sig["n_esn_better"] + sig["n_qrc_better"] + sig["n_tied"] == 3


class TestRealFinancialExecutionBlocked:
    def test_run_real_financial_task_raises(self):
        with pytest.raises(RealFinancialExecutionBlocked):
            run_real_financial_task()

    def test_blocked_reason_cites_readiness_gate(self):
        with pytest.raises(RealFinancialExecutionBlocked) as excinfo:
            run_real_financial_task("SPY", "QQQ")
        msg = str(excinfo.value)
        assert "READINESS_BRIEF.md" in msg
        assert "RESEARCH_ONLY" in msg
        assert "SPY" in msg and "USO" in msg
