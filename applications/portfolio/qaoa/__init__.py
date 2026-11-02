"""QAOA-based portfolio optimization — re-exports the implementation from
``applications.portfolio.optimizer``.

This module exists so the generation-aware layout
(``applications/portfolio/{qaoa,classical,prototypes}/``) is navigable without
physically moving and re-testing the working module. The
implementation lives in one place; this is the versioned front door to it.
"""
from applications.portfolio.optimizer import (
    PortfolioOptimizationReport,
    PortfolioOptimizer,
    run_portfolio_optimization,
)

__all__ = [
    "PortfolioOptimizationReport",
    "PortfolioOptimizer",
    "run_portfolio_optimization",
]
