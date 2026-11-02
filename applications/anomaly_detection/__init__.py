"""Financial anomaly/fraud detection application, built on
`algorithms_extended.qrbf_ocsvm_anomaly` (documented replication failure vs.
classical baseline -- see that package's THIRD_PARTY_NOTICES.md before assuming this
gives a quantum advantage).

This module adds `general_anomaly.py`: the same quantum/classical comparison,
generalized beyond fraud transactions to any binary-anomaly
`FinancialDataset` (e.g. `synthetic_trade_anomalies
.SyntheticTradePositionDataset`), routed through the unified
`applications.classification.dispatcher` rather than a bespoke pipeline."""
from applications.anomaly_detection.fraud_detector import FraudDetectionApplication, FraudDetectionReport, detect_anomalies
from applications.anomaly_detection.general_anomaly import GeneralAnomalyReport, detect_general_anomalies
from applications.anomaly_detection.synthetic_trade_anomalies import SyntheticTradePositionDataset
from applications.anomaly_detection.synthetic_transactions import SyntheticTransactionDataset

__all__ = [
    "FraudDetectionApplication", "FraudDetectionReport", "detect_anomalies",
    "GeneralAnomalyReport", "detect_general_anomalies",
    "SyntheticTransactionDataset", "SyntheticTradePositionDataset",
]
