"""End-to-end example: a real 5-way (6 with quantum) VaR/CVaR comparison on
one synthetic, fat-tailed P&L sample, plus a VaR-breach backtest and a
scenario-grid sweep through the same tail-risk pipeline. Run:

    python -m examples.tail_risk_multi_method_comparison_example
"""
from __future__ import annotations

import numpy as np

from applications.scenario_analysis.generator import default_scenario_grid
from applications.scenario_analysis.tail_risk_adapter import run_tail_risk_scenarios
from applications.tail_risk.backtest import backtest_var_breaches
from applications.tail_risk.comparison import compare_methods
from applications.tail_risk.domain import TailRiskSpec
from applications.tail_risk.walk_forward import rolling_tail_risk


def _synthetic_fat_tailed_pnl(seed: int = 42, n: int = 6000) -> np.ndarray:
    """Student-t(df=4) daily P&L, scaled to ~1% daily vol with a small
    positive drift -- fat tails by construction, so the Gaussian/CF/
    historical/quantum divergence in the comparison below is real, not
    manufactured noise.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_t(df=4, size=n) * 0.01 + 0.0002


def main() -> None:
    pnl = _synthetic_fat_tailed_pnl()

    print("=" * 78)
    print("1. Multi-method VaR/CVaR comparison (alpha=0.95)")
    print("=" * 78)
    spec = TailRiskSpec(confidence_levels=[0.95], run_quantum=True, n_grid_qubits=3, epsilon=0.05, shots=1024, seed=42)
    report = compare_methods(pnl, spec)

    print(f"n_samples={report.n_samples}  prob_of_loss={report.prob_of_loss:.4f}")
    print(f"{'method':<28}{'VaR':>12}{'CVaR':>12}{'wall_time_s':>14}")
    for name, m in report.results.items():
        print(f"{name:<28}{m.var:>12.5f}{m.cvar:>12.5f}{m.wall_time_s:>14.4f}")

    print("\nDiagnostics:")
    for note in report.notes:
        print(f"  - {note}")

    print("\n" + "=" * 78)
    print("2. VaR-breach backtest: does the historical VaR estimate hold up?")
    print("=" * 78)
    var_hist = report.results["historical"].var
    breach = backtest_var_breaches(pnl, var_hist, alpha=0.95)
    print(
        f"breaches={breach.n_breaches}/{breach.n_obs} "
        f"(observed rate={breach.breach_rate:.4f}, expected={breach.expected_rate:.4f})"
    )
    print(f"Kupiec LR={breach.kupiec_lr_stat:.4f}  p-value={breach.kupiec_p_value:.4f}  reject_H0={breach.reject_at_95}")

    print("\n" + "=" * 78)
    print("3. Walk-forward: does the VaR estimate drift fold to fold?")
    print("=" * 78)
    wf = rolling_tail_risk(pnl, n_splits=5, alpha=0.95)
    for fold in wf.folds:
        bt = fold.breach_test
        print(
            f"  fold {fold.fold}: train={fold.n_train:5d} test={fold.n_test:5d} "
            f"VaR={fold.var_estimate:.5f} CVaR={fold.cvar_estimate:.5f} "
            f"breaches={bt.n_breaches}/{bt.n_obs} kupiec_p={bt.kupiec_p_value:.4f}"
        )

    print("\n" + "=" * 78)
    print("4. Scenario grid: the same tail-risk comparison under real stress shocks")
    print("=" * 78)
    scenarios = default_scenario_grid()
    scenario_spec = TailRiskSpec(confidence_levels=[0.95], run_quantum=False, seed=42)
    grid_report = run_tail_risk_scenarios(pnl, scenarios, spec=scenario_spec, seed=42)
    print(f"{'scenario':<22}{'hist VaR':>12}{'hist CVaR':>12}{'CF VaR':>12}")
    for row in grid_report.rows:
        hist = row.risk.results["historical"]
        cf = row.risk.results["cornish_fisher"]
        print(f"{row.scenario:<22}{hist.var:>12.5f}{hist.cvar:>12.5f}{cf.var:>12.5f}")


if __name__ == "__main__":
    main()
