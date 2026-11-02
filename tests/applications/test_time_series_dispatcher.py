import numpy as np
import pytest

from applications.time_series import methods as method_adapters
from applications.time_series.dispatcher import ARXIV_GATE_NOTE, run_dispatcher
from applications.time_series.domain import (
    MethodChoice,
    QRCHyperparams,
    TargetKind,
    TimeSeriesForecastSpec,
    ValidationStrategy,
)
from applications.time_series.errors import UnknownForecastMethodError
from applications.time_series.evaluation import walk_forward_folds
from applications.time_series.synthetic import SyntheticVolatilityDataset

# Reduced-scale reservoir/config for test speed (documented, not silently
# hidden) -- small qubit counts and few splits. The comparison logic under
# test does not depend on the reservoirs being large enough to be a good
# forecaster, only on the dispatcher wiring being correct.
_FAST_QRC = QRCHyperparams(n1=3, n2=2, arxiv_window=3)


def _all_methods_spec(**overrides) -> TimeSeriesForecastSpec:
    kwargs = dict(
        target_kind=TargetKind.REALIZED_VOLATILITY,
        methods=list(MethodChoice),
        validation=ValidationStrategy.WALK_FORWARD,
        n_splits=3,
        qrc=_FAST_QRC,
        seed=7,
    )
    kwargs.update(overrides)
    return TimeSeriesForecastSpec(**kwargs)


def test_dispatcher_runs_every_requested_method_through_one_interface():
    dataset = SyntheticVolatilityDataset(n_steps=250, seed=7)
    spec = _all_methods_spec()
    result = run_dispatcher(dataset, spec, log_experiments=False)

    assert set(result.results.keys()) == {m.value for m in MethodChoice}
    # every non-naive method actually went through feature extraction + a
    # ridge readout (not a mock): each has fold-level metrics or an honestly
    # captured failure, never silently absent.
    for name, r in result.results.items():
        assert r.failed or len(r.fold_metrics) > 0
        if not r.failed:
            assert np.isfinite(r.mean_mse)
            assert np.isfinite(r.mean_mae)


def test_best_method_is_actually_the_minimum_mse():
    dataset = SyntheticVolatilityDataset(n_steps=250, seed=7)
    spec = _all_methods_spec()
    result = run_dispatcher(dataset, spec, log_experiments=False)

    candidates = {k: r.mean_mse for k, r in result.results.items() if not r.failed and np.isfinite(r.mean_mse)}
    assert result.best_method in candidates
    assert candidates[result.best_method] == min(candidates.values())


def test_arxiv_gate_note_is_surfaced_only_when_that_method_is_requested():
    dataset = SyntheticVolatilityDataset(n_steps=200, seed=3)
    spec_with = _all_methods_spec(methods=[MethodChoice.QRC_GEN1_ARXIV, MethodChoice.NAIVE_CONSTANT])
    spec_without = _all_methods_spec(methods=[MethodChoice.ESN, MethodChoice.NAIVE_CONSTANT])

    result_with = run_dispatcher(dataset, spec_with, log_experiments=False)
    result_without = run_dispatcher(dataset, spec_without, log_experiments=False)

    assert result_with.arxiv_gate_note == ARXIV_GATE_NOTE
    assert "GATE_PASSED=False" in result_with.arxiv_gate_note
    assert result_without.arxiv_gate_note is None


def test_naive_floor_matches_naive_constant_result():
    dataset = SyntheticVolatilityDataset(n_steps=200, seed=3)
    spec = _all_methods_spec(methods=[MethodChoice.ESN, MethodChoice.NAIVE_CONSTANT])
    result = run_dispatcher(dataset, spec, log_experiments=False)
    assert result.naive_floor_mse == result.results["naive_constant"].mean_mse


def test_at_least_one_learned_method_beats_naive_constant():
    """Real assertion, not a trivial constant: on this synthetic dataset, at
    least one learned method must beat predicting the training mean for
    every point -- otherwise nothing here learned anything real."""
    dataset = SyntheticVolatilityDataset(n_steps=300, seed=7)
    spec = _all_methods_spec(methods=[MethodChoice.QRC_GEN2, MethodChoice.ESN, MethodChoice.MAMBA_SSM, MethodChoice.NAIVE_CONSTANT])
    result = run_dispatcher(dataset, spec, log_experiments=False)
    naive_mse = result.results["naive_constant"].mean_mse
    learned = [r.mean_mse for name, r in result.results.items() if name != "naive_constant" and not r.failed]
    assert learned, "no learned method produced a result to compare"
    assert min(learned) < naive_mse


def test_walk_forward_folds_never_leak_future_data():
    folds = walk_forward_folds(n=120, n_splits=5, min_train_fraction=0.4)
    assert len(folds) == 5
    prev_test_max = -1
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()
        assert test_idx.min() > prev_test_max  # non-overlapping, strictly advancing test windows
        prev_test_max = test_idx.max()


def test_unknown_method_raises_specific_error():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)
    spec = _all_methods_spec()
    # NAIVE_CONSTANT has no feature-extraction adapter by design (it is
    # special-cased in evaluation.py) -- calling extract_features with it
    # directly must raise the specific, documented exception.
    with pytest.raises(UnknownForecastMethodError):
        method_adapters.extract_features(MethodChoice.NAIVE_CONSTANT, X, y, spec)


def test_gen2_and_gen2_original_agree_as_a_refactor_parity_check():
    """QRC_GEN2 and QRC_GEN2_ORIGINAL run the same math through two
    different code paths (the current reusable class vs. the preserved
    pre-refactor module) -- they should produce identical results given
    identical hyperparameters/seed, which is itself a useful regression
    check on the Gen2 refactor."""
    dataset = SyntheticVolatilityDataset(n_steps=200, seed=5)
    spec = _all_methods_spec(methods=[MethodChoice.QRC_GEN2, MethodChoice.QRC_GEN2_ORIGINAL])
    result = run_dispatcher(dataset, spec, log_experiments=False)
    a, b = result.results["qrc_gen2"], result.results["qrc_gen2_original"]
    assert not a.failed and not b.failed
    assert a.mean_mse == pytest.approx(b.mean_mse, rel=1e-8)
