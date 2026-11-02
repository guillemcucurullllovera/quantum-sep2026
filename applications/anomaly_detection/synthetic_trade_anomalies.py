"""Synthetic trade/position anomaly dataset -- generalizes anomaly detection
in this platform beyond the Sparkov-style fraud-transaction shape that
`synthetic_transactions.py` was built around (see that module's docstring:
it exists because the real Sparkov CSV isn't shipped in-repo). A production
desk's anomaly-detection need is at least as often about its OWN book --
an unusual position size, an outsized single-trade P&L swing, a leverage
spike, a concentration breach -- as about incoming transaction fraud, and
those features have a genuinely different shape (portfolio/risk-desk
statistics, not card-transaction statistics), so this is a distinct
generator, not a renamed copy of `SyntheticTransactionDataset`.
"""
from __future__ import annotations

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.schemas import DatasetDescriptor


class SyntheticTradePositionDataset(FinancialDataset):
    """12-feature synthetic trade/position book: per-trade/position
    statistics (size z-score, P&L z-score, leverage ratio, turnover,
    concentration, holding period, venue/counterparty diversity, etc.),
    with injected anomalies representing risk-limit-breach-style events
    (a structurally different subset of features shifted hard -- oversized
    position + outsized leverage + low diversification, mimicking a rogue
    or limit-breaching trade rather than a uniform outlier)."""

    FEATURE_NAMES = (
        "position_size_zscore", "pnl_zscore", "leverage_ratio", "turnover_ratio",
        "concentration_hhi", "holding_period_days", "venue_diversity", "counterparty_diversity",
        "var_utilization", "margin_utilization", "realized_vol_zscore", "order_to_trade_ratio",
    )

    def __init__(
        self, n_normal: int = 400, n_anomaly: int = 60, anomaly_shift: float = 3.0, seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        n_features = len(self.FEATURE_NAMES)

        normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
        # keep leverage/margin/VaR-utilization non-negative-ish, matching how
        # these ratios actually behave on a real book
        for col in (2, 8, 9):
            normal[:, col] = np.abs(normal[:, col]) * 0.5 + 0.3

        anomaly = rng.normal(loc=0.0, scale=1.0, size=(n_anomaly, n_features))
        for col in (2, 8, 9):
            anomaly[:, col] = np.abs(anomaly[:, col]) * 0.5 + 0.3
        # a risk-limit-breach anomaly hits position size, leverage, and VaR/margin
        # utilization hard, and diversification (venue/counterparty) low --
        # a structurally coherent anomaly pattern, not an arbitrary uniform shift
        breach_cols = [0, 2, 8, 9]
        anomaly[:, breach_cols] += anomaly_shift * rng.choice([1.0], size=(n_anomaly, len(breach_cols)))
        diversity_cols = [6, 7]
        anomaly[:, diversity_cols] -= anomaly_shift * 0.6

        X = np.vstack([normal, anomaly])
        y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)])
        perm = rng.permutation(len(y))
        self._X, self._y = X[perm], y[perm]
        self._n_features = n_features

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="synthetic_trade_position_anomaly", n_samples=len(self._y),
            n_features=self._n_features, domain="trade_position_anomaly_detection",
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y


__all__ = ["SyntheticTradePositionDataset"]
