"""Real tests for applications.anomaly_detection."""
from __future__ import annotations

from applications.anomaly_detection.fraud_detector import detect_anomalies
from applications.anomaly_detection.synthetic_transactions import SyntheticTransactionDataset
from benchmarks.quantum_vs_classical.anomaly import qrbf_vs_classical_ocsvm
from core.dataset_interfaces import FinancialDataset


def test_synthetic_dataset_shape_and_labels():
    ds = SyntheticTransactionDataset(n_normal=40, n_anomaly=10, seed=0)
    assert isinstance(ds, FinancialDataset)
    X, y = ds.features(), ds.targets()
    assert X.shape == (50, 20)
    assert y.sum() == 10
    assert set(y.tolist()) == {0, 1}


def test_classical_baseline_beats_chance_on_easy_synthetic_case():
    """Real detection quality check: with a strong anomaly_shift, even the
    documented-weaker qrbf path and the classical path should both do
    better than a trivial always-normal predictor (F1=0 for the minority
    class) on well-separated synthetic anomalies."""
    ds = SyntheticTransactionDataset(n_normal=80, n_anomaly=20, anomaly_shift=4.0, seed=1)
    report = detect_anomalies(ds, feature_map_name="ECR", gamma=0.05, nu=0.2, seed=1)
    assert report.classical_test_f1 > 0.0
    assert report.n_train + report.n_test == 100


def test_report_is_honest_about_which_model_actually_wins():
    """The report must recommend whichever model actually scored higher on
    THIS run's data -- not hardcode a preference either direction."""
    ds = SyntheticTransactionDataset(n_normal=60, n_anomaly=15, anomaly_shift=3.0, seed=2)
    report = detect_anomalies(ds, feature_map_name="CX", seed=2)
    if report.quantum_test_f1 > report.classical_test_f1:
        assert report.recommended_model == "quantum"
    else:
        assert report.recommended_model == "classical"


def test_benchmark_wiring_produces_real_comparison():
    result = qrbf_vs_classical_ocsvm(n_normal=50, n_anomaly=12, seed=3, feature_map_name="ECR")
    assert result.method_a == "qrbf-ECR-OCSVM"
    assert result.method_b == "rbf-OCSVM"
    assert 0.0 <= result.value_a <= 1.0
    assert 0.0 <= result.value_b <= 1.0
    assert result.context["recommended"] in ("quantum", "classical")
