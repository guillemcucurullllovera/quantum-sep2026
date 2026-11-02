import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from applications.time_series.domain import FeatureSpec, NoiseSpec, TimeSeriesForecastSpec
from applications.time_series.errors import InsufficientDataError
from applications.time_series.preprocessing import ArrayDataset, apply_fracdiff, apply_horizon, build_pipeline


def test_feature_spec_rejects_out_of_range_fracdiff_d():
    with pytest.raises(PydanticValidationError):
        FeatureSpec(fracdiff_d=2.5)


def test_noise_spec_rejects_out_of_range_probability():
    with pytest.raises(PydanticValidationError):
        NoiseSpec(depolarizing_prob_1q=1.5)


def test_forecast_spec_rejects_zero_horizon():
    with pytest.raises(PydanticValidationError):
        TimeSeriesForecastSpec(horizon=0)


def test_forecast_spec_rejects_zero_splits():
    with pytest.raises(PydanticValidationError):
        TimeSeriesForecastSpec(n_splits=0)


def test_array_dataset_walk_forward_split_is_leakage_safe():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = rng.standard_normal(100)
    ds = ArrayDataset(X, y, name="test")
    for train_idx, test_idx in ds.split_walk_forward(n_splits=4):
        assert train_idx.max() < test_idx.min()


def test_apply_fracdiff_noop_when_disabled():
    X = np.arange(20.0).reshape(10, 2)
    spec = FeatureSpec(use_fracdiff=False)
    out = apply_fracdiff(X, spec)
    np.testing.assert_array_equal(out, X)


def test_apply_fracdiff_shrinks_and_is_finite():
    rng = np.random.default_rng(1)
    X = np.cumsum(rng.standard_normal((200, 3)), axis=0)  # non-stationary random walk columns
    spec = FeatureSpec(use_fracdiff=True, fracdiff_d=0.4, fracdiff_threshold=1e-3)
    out = apply_fracdiff(X, spec)
    assert out.shape[1] == X.shape[1]
    assert out.shape[0] < X.shape[0]  # warm-up rows trimmed
    assert np.isfinite(out).all()


def test_apply_fracdiff_raises_when_series_too_short():
    X = np.arange(6.0).reshape(3, 2)
    spec = FeatureSpec(use_fracdiff=True, fracdiff_d=0.4, fracdiff_threshold=1e-8)
    with pytest.raises(InsufficientDataError):
        apply_fracdiff(X, spec)


def test_apply_horizon_shifts_target_forward_without_lookback():
    X = np.arange(10.0).reshape(10, 1)
    y = np.arange(100.0, 110.0)
    X_h, y_h = apply_horizon(X, y, horizon=3)
    assert len(X_h) == len(y_h) == 8
    # y_h[t] must equal y[t + horizon - 1] for every t -- forward-only shift
    for t in range(len(y_h)):
        assert y_h[t] == y[t + 2]


def test_apply_horizon_noop_for_horizon_one():
    X = np.arange(10.0).reshape(10, 1)
    y = np.arange(100.0, 110.0)
    X_h, y_h = apply_horizon(X, y, horizon=1)
    np.testing.assert_array_equal(X_h, X)
    np.testing.assert_array_equal(y_h, y)


def test_build_pipeline_composes_fracdiff_then_horizon():
    rng = np.random.default_rng(2)
    X = np.cumsum(rng.standard_normal((150, 2)), axis=0)
    y = rng.standard_normal(150)
    spec = FeatureSpec(use_fracdiff=True, fracdiff_d=0.4, fracdiff_threshold=1e-3)
    X_out, y_out = build_pipeline(X, y, spec, horizon=2)
    assert len(X_out) == len(y_out)
    assert len(X_out) < len(X)
