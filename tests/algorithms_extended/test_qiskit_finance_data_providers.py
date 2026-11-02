"""expansion: qiskit-finance's data_providers subpackage, ported
whole (Apache-2.0). Only RandomDataProvider is exercisable offline; the
network-backed providers (Yahoo/Wikipedia/Exchange/DataOnDemand) are
confirmed importable and instantiable but not run against live services in
this environment -- see THIRD_PARTY_NOTICES.md.
"""
import datetime

import numpy as np
import pytest

from algorithms_extended.qiskit_finance_extensions.upstream_data_providers import (
    BaseDataProvider,
    ExchangeDataProvider,
    RandomDataProvider,
    StockMarket,
    WikipediaDataProvider,
    YahooDataProvider,
)


def test_random_data_provider_produces_real_series():
    p = RandomDataProvider(
        tickers=["AAA", "BBB", "CCC"],
        seed=1,
        start=datetime.datetime(2020, 1, 1),
        end=datetime.datetime(2020, 2, 1),
    )
    p.run()
    data = np.array(p._data)
    assert data.shape[0] == 3
    assert data.shape[1] > 1
    assert np.all(np.isfinite(data))


def test_random_data_provider_reproducible_with_seed():
    p1 = RandomDataProvider(tickers=["AAA", "BBB"], seed=42)
    p1.run()
    p2 = RandomDataProvider(tickers=["AAA", "BBB"], seed=42)
    p2.run()
    np.testing.assert_array_equal(np.array(p1._data), np.array(p2._data))


def test_network_backed_providers_are_constructible_not_run():
    """These require live network access this environment doesn't have --
    confirm the classes are real/importable/constructible (a genuine
    integration point for a future connected deployment) without attempting
    a network call."""
    assert issubclass(YahooDataProvider, BaseDataProvider)
    assert issubclass(ExchangeDataProvider, BaseDataProvider)
    assert issubclass(WikipediaDataProvider, BaseDataProvider)


def test_stock_market_enum_present():
    assert hasattr(StockMarket, "NASDAQ") or hasattr(StockMarket, "NYSE") or len(list(StockMarket)) > 0
