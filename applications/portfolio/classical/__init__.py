"""Classical portfolio-optimization comparator, re-exported here so the
QAOA-vs-classical generations sit side by side under
``applications/portfolio/``. Implementation lives in ``classical.portfolio``.
"""
from classical.portfolio.markowitz import (
    equal_weight_baseline,
    markowitz_min_variance,
    portfolio_variance,
)

__all__ = ["equal_weight_baseline", "markowitz_min_variance", "portfolio_variance"]
