"""
Triple-Barrier Label Generator.

Reference: López de Prado (2018) "Advances in Financial Machine Learning"
           Chapter 3 — Labeling.

Labels each bar (time step) by whichever barrier is touched first:
    +1 (buy)  : upper profit-taking barrier (pt) touched first
    -1 (sell) : lower stop-loss barrier (sl) touched first
     0 (hold) : neither barrier touched before time horizon t1

This is the correct labeling method for directional financial ML tasks.
Standard fixed-window return labels are biased when return variance is
non-stationary (i.e. always in financial data).

No dependencies beyond numpy and pandas.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def get_daily_vol(
    close: pd.Series,
    span: int = 100,
) -> pd.Series:
    """
    Daily volatility estimate using exponentially weighted std of log returns.
    Used to set pt/sl barriers relative to local volatility.
    """
    log_ret = np.log(close).diff().dropna()
    return log_ret.ewm(span=span).std()


def triple_barrier_labels(
    close: pd.Series,
    events: pd.Series,
    pt: float = 0.02,
    sl: float = 0.01,
    t1: Optional[pd.Series] = None,
    min_ret: float = 0.0,
    dynamic_width: bool = False,
    vol: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Apply triple-barrier method to label financial events.

    Args:
        close         : adjusted close price series
        events        : pd.Series of event timestamps (index = event start times)
        pt            : profit-taking barrier width as fraction of price
                        (e.g. 0.02 = 2%); OR multiple of daily vol if dynamic_width=True
        sl            : stop-loss barrier width (same units as pt)
        t1            : pd.Series of event end times (time horizon per event);
                        if None, uses a fixed 20-bar window
        min_ret       : minimum absolute return to assign a non-zero label
                        (events with |return| < min_ret get label 0)
        dynamic_width : if True, pt/sl are multiples of daily vol (requires vol)
        vol           : daily vol series (required if dynamic_width=True)
    Returns:
        pd.Series of labels {1, -1, 0} aligned to events index
    """
    if dynamic_width and vol is None:
        raise ValueError("vol is required when dynamic_width=True")

    close = close.sort_index()
    labels = {}

    for t0 in events.index:
        if t0 not in close.index:
            continue

        # Determine time horizon
        if t1 is not None and t0 in t1.index:
            t_end = t1[t0]
        else:
            loc = close.index.get_loc(t0)
            end_loc = min(loc + 20, len(close) - 1)
            t_end = close.index[end_loc]

        price_path = close.loc[t0:t_end]
        if len(price_path) < 2:
            labels[t0] = 0
            continue

        p0 = price_path.iloc[0]
        returns = (price_path - p0) / p0

        # Set barrier widths
        if dynamic_width:
            v = float(vol.loc[t0]) if t0 in vol.index else 0.01
            upper = pt * v
            lower = -sl * v
        else:
            upper = pt
            lower = -sl

        # Find first barrier touched
        upper_touch = returns[returns >= upper]
        lower_touch = returns[returns <= lower]

        t_upper = upper_touch.index[0] if len(upper_touch) > 0 else pd.NaT
        t_lower = lower_touch.index[0] if len(lower_touch) > 0 else pd.NaT

        if pd.isna(t_upper) and pd.isna(t_lower):
            # Neither barrier touched — label by final return direction
            final_ret = float(returns.iloc[-1])
            if abs(final_ret) < min_ret:
                labels[t0] = 0
            else:
                labels[t0] = 1 if final_ret > 0 else -1
        elif pd.isna(t_lower) or (not pd.isna(t_upper) and t_upper <= t_lower):
            labels[t0] = 1     # upper (profit-taking) touched first
        else:
            labels[t0] = -1    # lower (stop-loss) touched first

    return pd.Series(labels, dtype=int, name="triple_barrier_label")


def events_from_cusum(
    close: pd.Series,
    threshold: float,
) -> pd.Series:
    """
    Generate event timestamps using the symmetric CUSUM filter.

    Returns a Series of timestamps where the cumulative sum of log returns
    crosses ±threshold — used as entry points for triple_barrier_labels.

    Reference: López de Prado (2018) Chapter 2.
    """
    log_ret = np.log(close).diff().dropna()
    s_pos, s_neg = 0.0, 0.0
    timestamps = []

    for t, r in log_ret.items():
        s_pos = max(0, s_pos + r)
        s_neg = min(0, s_neg + r)
        if s_pos >= threshold:
            s_pos = 0.0
            timestamps.append(t)
        elif s_neg <= -threshold:
            s_neg = 0.0
            timestamps.append(t)

    if not timestamps:
        return pd.Series([], dtype="datetime64[ns]")
    return pd.Series(timestamps, index=timestamps)


def label_stats(labels: pd.Series) -> dict:
    """Return class distribution statistics for a label series."""
    counts = labels.value_counts().to_dict()
    total = len(labels)
    return {
        "total": total,
        "buy": counts.get(1, 0),
        "hold": counts.get(0, 0),
        "sell": counts.get(-1, 0),
        "pct_buy": counts.get(1, 0) / total if total else 0,
        "pct_hold": counts.get(0, 0) / total if total else 0,
        "pct_sell": counts.get(-1, 0) / total if total else 0,
    }


__all__ = [
    "triple_barrier_labels",
    "events_from_cusum",
    "get_daily_vol",
    "label_stats",
]
