"""Tests for `applications.regime_detection.regime_product`: the
confidence-routed HMM/classifier ensemble."""
from __future__ import annotations

import numpy as np
import pytest

from applications.classification.errors import UnsupportedTaskError
from applications.classification.task_spec import Algorithm
from applications.regime_detection.regime_product import run_regime_ensemble
from applications.regime_detection.synthetic_regime_dataset import SyntheticRegimeDataset


def test_regime_ensemble_runs_and_reports_all_sources():
    ds = SyntheticRegimeDataset(n_per_regime=50, n_features=5, seed=11)
    result = run_regime_ensemble(
        ds, classifier_algorithm=Algorithm.VQC_V5_REUPLOAD, n_qubits=5, seed=1, confidence_threshold=0.6
    )
    assert 0.0 <= result.ensemble_accuracy <= 1.0
    assert 0.0 <= result.hmm_accuracy <= 1.0
    assert 0.0 <= result.classifier_accuracy <= 1.0
    assert 0.0 <= result.frac_routed_to_hmm <= 1.0
    assert set(np.unique(result.source)) <= {"hmm", "vqc_v5_reupload"}
    assert len(result.predictions) == len(result.source)


def test_confidence_threshold_controls_routing():
    ds = SyntheticRegimeDataset(n_per_regime=40, n_features=5, seed=12)
    # threshold below any possible posterior probability (posteriors are in
    # [1/3, 1]) -- everything routes to HMM
    always_hmm = run_regime_ensemble(
        ds, classifier_algorithm=Algorithm.CLASSICAL_LOGISTIC_REGRESSION, n_qubits=4, seed=2, confidence_threshold=0.0
    )
    assert always_hmm.frac_routed_to_hmm == 1.0
    assert np.all(always_hmm.source == "hmm")

    # threshold above any achievable posterior (max is 1.0, so >1.0 is unreachable)
    never_hmm = run_regime_ensemble(
        ds, classifier_algorithm=Algorithm.CLASSICAL_LOGISTIC_REGRESSION, n_qubits=4, seed=2, confidence_threshold=1.01
    )
    assert never_hmm.frac_routed_to_hmm == 0.0
    assert np.all(never_hmm.source == "classical_logistic_regression")
    # when a source is never used, its "only" accuracy is reported as NaN, not 0/hidden
    assert np.isnan(never_hmm.hmm_only_accuracy)


def test_regime_ensemble_requires_exactly_three_regimes():
    from applications.classification.synthetic_financial_dataset import SyntheticDirectionalDataset

    ds = SyntheticDirectionalDataset(n_samples=40, seed=1)  # binary, not 3-regime
    with pytest.raises(UnsupportedTaskError):
        run_regime_ensemble(ds, n_qubits=4)
