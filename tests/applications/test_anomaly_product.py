"""Tests for the general-purpose anomaly detection product layer
(`applications.anomaly_detection.general_anomaly` +
`synthetic_trade_anomalies`) -- the fraud-detection pattern generalized to a
non-fraud finance context, routed through the same dispatcher used by
classification and regime detection."""
from __future__ import annotations

from applications.anomaly_detection.general_anomaly import detect_general_anomalies
from applications.anomaly_detection.synthetic_trade_anomalies import SyntheticTradePositionDataset
from core.dataset_interfaces import FinancialDataset


def test_synthetic_trade_dataset_shape_and_labels():
    ds = SyntheticTradePositionDataset(n_normal=60, n_anomaly=15, seed=0)
    assert isinstance(ds, FinancialDataset)
    X, y = ds.features(), ds.targets()
    assert X.shape == (75, 12)
    assert y.sum() == 15
    assert set(y.tolist()) == {0, 1}


def test_general_anomaly_detection_runs_through_dispatcher_and_reports_honestly():
    ds = SyntheticTradePositionDataset(n_normal=60, n_anomaly=15, anomaly_shift=3.5, seed=1)
    report = detect_general_anomalies(ds, seed=1)
    assert report.dataset_name == "synthetic_trade_position_anomaly"
    assert 0.0 <= report.quantum_f1 <= 1.0
    assert 0.0 <= report.classical_f1 <= 1.0
    assert report.n_train + report.n_test == 75
    if report.quantum_f1 > report.classical_f1:
        assert report.recommended_model == "quantum"
    else:
        assert report.recommended_model == "classical"
    # both results came from the same dispatcher, same ANOMALY task type
    assert report.quantum_result.task_type == report.classical_result.task_type


def test_general_anomaly_detection_works_on_transaction_dataset_too():
    """Same product function, different (fraud-shaped) dataset -- proves this
    is genuinely dataset-agnostic, not hardcoded to trade/position shape."""
    from applications.anomaly_detection.synthetic_transactions import SyntheticTransactionDataset

    ds = SyntheticTransactionDataset(n_normal=60, n_anomaly=15, seed=2)
    report = detect_general_anomalies(ds, seed=2)
    assert 0.0 <= report.quantum_f1 <= 1.0
    assert 0.0 <= report.classical_f1 <= 1.0
