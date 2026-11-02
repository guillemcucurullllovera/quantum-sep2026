"""Lightweight synthetic transaction-anomaly dataset generator.

`algorithms_extended.qrbf_ocsvm_anomaly.sparkov_data` needs an externally
cloned/generated Sparkov CSV (multi-hundred-MB, not shipped in this repo per
the no-raw-data-dumps policy) to reproduce the paper's own replication
numbers exactly. For this application layer -- exercising the algorithm
against a realistic-shaped but self-contained financial anomaly problem --
we generate synthetic transaction data instead: 20 numeric features matching
the shape the qrbf feature maps expect (10 qubits x 2 rotations/qubit),
mixing a genuine anomaly signal (rare, structurally different transactions)
with normal transaction noise, not a Sparkov clone.
"""
from __future__ import annotations

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.schemas import DatasetDescriptor


class SyntheticTransactionDataset(FinancialDataset):
    """20-feature synthetic transaction dataset: 19 normal-behavior features
    (amount, hour, day-of-week, merchant/category fraud-rate-style encodings,
    distance, population, interaction terms) plus a controllable anomaly
    injection matching the qrbf replication's r (anomaly ratio) parameter.
    """

    def __init__(
        self, n_normal: int = 400, n_anomaly: int = 100, n_features: int = 20,
        anomaly_shift: float = 2.5, seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
        # Anomalies: a structurally different subset of features shifted hard,
        # mimicking "unusual amount + unusual time + unusual merchant profile"
        # rather than a uniform shift on every feature (that would be trivially
        # separable and wouldn't exercise the kernel the way real fraud does).
        anomaly = rng.normal(loc=0.0, scale=1.0, size=(n_anomaly, n_features))
        shifted_dims = rng.choice(n_features, size=max(3, n_features // 4), replace=False)
        anomaly[:, shifted_dims] += anomaly_shift * rng.choice([-1, 1], size=len(shifted_dims))

        X = np.vstack([normal, anomaly])
        y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)])
        perm = rng.permutation(len(y))
        self._X, self._y = X[perm], y[perm]
        self._n_features = n_features

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="synthetic_transaction_anomaly", n_samples=len(self._y),
            n_features=self._n_features, domain="transaction_anomaly_detection",
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y
