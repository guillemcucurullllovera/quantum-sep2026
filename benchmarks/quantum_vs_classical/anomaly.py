"""qrbf-OCSVM vs classical RBF-OCSVM, on the same synthetic transaction-
anomaly dataset -- the honest quantum-vs-classical comparison for
`algorithms_extended.qrbf_ocsvm_anomaly` (documented replication failure;
classical is expected to win here, and this benchmark says so plainly rather
than dressing it up)."""
from __future__ import annotations

from applications.anomaly_detection.fraud_detector import detect_anomalies
from applications.anomaly_detection.synthetic_transactions import SyntheticTransactionDataset
from benchmarks.runner import MethodOutput, run_benchmark
from core.schemas import BenchmarkResult


def qrbf_vs_classical_ocsvm(
    n_normal: int = 400, n_anomaly: int = 100, seed: int = 0,
    feature_map_name: str = "ECR",
) -> BenchmarkResult:
    dataset = SyntheticTransactionDataset(n_normal=n_normal, n_anomaly=n_anomaly, seed=seed)
    report = detect_anomalies(dataset, feature_map_name=feature_map_name, seed=seed)

    def quantum() -> MethodOutput:
        return MethodOutput(value=report.quantum_test_f1)

    def classical() -> MethodOutput:
        return MethodOutput(value=report.classical_test_f1)

    return run_benchmark(
        method_a=quantum, method_b=classical,
        method_a_name=f"qrbf-{feature_map_name}-OCSVM", method_b_name="rbf-OCSVM",
        metric_name="anomaly_detection_test_f1", seed=seed,
        context={"n_normal": n_normal, "n_anomaly": n_anomaly, "recommended": report.recommended_model},
    )
