import numpy as np
import pandas as pd

from tools.dataset_preparation import prepare


def test_prepare_produces_stationary_and_labeled_output():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    # random-walk-like price series (non-stationary), long enough for fracdiff+labeling
    returns = rng.normal(0, 0.01, n)
    prices = pd.Series(100 * np.cumprod(1 + returns), index=idx)

    out = prepare(prices, max_holding=5)

    assert "price" in out.columns
    assert "stationary" in out.columns
    assert "label" in out.columns
    assert len(out) == n
    assert out.attrs["fracdiff_d"] > 0.0
    # labels should be a real mix, not all-zero (would indicate a wiring bug)
    non_null_labels = out["label"].dropna()
    assert len(non_null_labels) > 0
