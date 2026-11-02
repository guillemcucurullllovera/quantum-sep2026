"""VaR/CVaR/probability-of-loss estimation: a real multi-method comparison
(historical, parametric Gaussian, Cornish-Fisher, classical Monte Carlo GBM,
the meta-agent regime-conditional bootstrap, and quantum amplitude
estimation) always reported side by side, plus VaR-breach backtesting and
walk-forward evaluation.
"""
from applications.tail_risk.backtest import BreachTestResult, backtest_var_breaches, count_var_breaches, kupiec_pof_test
from applications.tail_risk.comparison import compare_methods, compare_methods_all_levels
from applications.tail_risk.domain import (
    DistributionAssumption,
    MethodResult,
    PositionSpec,
    TailRiskComparisonReport,
    TailRiskSpec,
)
from applications.tail_risk.errors import DegenerateTailError, EmptySampleError, validate_pnl_sample
from applications.tail_risk.integration import benchmark_qae_vs_historical, log_comparison
from applications.tail_risk.var_cvar import TailRiskReport, assess_tail_risk
from applications.tail_risk.walk_forward import WalkForwardFold, WalkForwardReport, rolling_tail_risk

__all__ = [
    "TailRiskReport",
    "assess_tail_risk",
    "TailRiskSpec",
    "PositionSpec",
    "DistributionAssumption",
    "MethodResult",
    "TailRiskComparisonReport",
    "compare_methods",
    "compare_methods_all_levels",
    "EmptySampleError",
    "DegenerateTailError",
    "validate_pnl_sample",
    "BreachTestResult",
    "backtest_var_breaches",
    "count_var_breaches",
    "kupiec_pof_test",
    "WalkForwardFold",
    "WalkForwardReport",
    "rolling_tail_risk",
    "log_comparison",
    "benchmark_qae_vs_historical",
]
