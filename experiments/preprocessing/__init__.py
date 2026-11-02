"""Composable preprocessing pipeline built on core.feature_engineering.

This module does not reimplement fracdiff/purged-CV/triple-barrier (those
live in core.feature_engineering) — it composes
them into an ordered pipeline so an experiment config can specify
`preprocessing: [fracdiff, triple_barrier]` instead of every caller hand-
wiring the same three-line glue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.feature_engineering.fracdiff import fracdiff
from core.feature_engineering.triple_barrier import get_daily_vol, triple_barrier_labels

Step = Callable[[pd.Series], pd.Series]


@dataclass
class PreprocessingStep:
    name: str
    fn: Step
    params: dict[str, Any] = field(default_factory=dict)

    def apply(self, series: pd.Series) -> pd.Series:
        return self.fn(series, **self.params)


def _fracdiff_step(series: pd.Series, d: float = 0.4, threshold: float = 1e-4) -> pd.Series:
    return fracdiff(series, d=d, threshold=threshold)


def _zscore_step(series: pd.Series, window: int = 60) -> pd.Series:
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    return (series - roll_mean) / roll_std.replace(0.0, np.nan)


def _clip_outliers_step(series: pd.Series, n_std: float = 5.0) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    return series.clip(lower=mu - n_std * sigma, upper=mu + n_std * sigma)


STEP_REGISTRY: dict[str, Step] = {
    "fracdiff": _fracdiff_step,
    "zscore": _zscore_step,
    "clip_outliers": _clip_outliers_step,
}


@dataclass
class Pipeline:
    steps: list[PreprocessingStep]

    def run(self, series: pd.Series) -> pd.Series:
        out = series
        for step in self.steps:
            out = step.apply(out)
        return out.dropna()

    @classmethod
    def from_spec(cls, spec: list[dict[str, Any]]) -> "Pipeline":
        """spec = [{"name": "fracdiff", "params": {"d": 0.3}}, {"name": "zscore"}]"""
        steps = []
        for item in spec:
            name = item["name"]
            if name not in STEP_REGISTRY:
                raise ValueError(f"Unknown preprocessing step '{name}'. Registered: {sorted(STEP_REGISTRY)}")
            steps.append(PreprocessingStep(name=name, fn=STEP_REGISTRY[name], params=item.get("params", {})))
        return cls(steps=steps)


def label_pipeline(
    prices: pd.Series, *, vol_window: int = 20, pt_mult: float = 1.0, sl_mult: float = 1.0, horizon: int = 10
) -> pd.Series:
    """End-to-end: daily-vol estimate -> triple-barrier labels. Composes
    core.feature_engineering.triple_barrier rather than reimplementing it.
    """
    vol = get_daily_vol(prices, span=vol_window)
    t_events = prices.index[vol_window:]
    t1 = pd.Series(prices.index[horizon:], index=prices.index[: len(prices) - horizon])
    return triple_barrier_labels(prices, t_events, vol, pt_sl=(pt_mult, sl_mult), t1=t1)
