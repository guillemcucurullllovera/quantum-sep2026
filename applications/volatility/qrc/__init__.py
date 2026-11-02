"""QRC-based realized-volatility forecasting -- re-exports from
``applications.volatility.realized_volatility``.
"""
from applications.volatility.realized_volatility import (
    PaperRealizedVolatilityDataset,
    compare_realized_volatility_methods,
    forecast_realized_volatility,
)

__all__ = [
    "PaperRealizedVolatilityDataset",
    "forecast_realized_volatility",
    "compare_realized_volatility_methods",
]
