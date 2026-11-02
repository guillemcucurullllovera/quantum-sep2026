"""Abstract interface every financial dataset loader in this platform must
implement, so algorithms/applications can be written against one contract
instead of each hand-rolling its own CSV-reading logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.schemas import DatasetDescriptor


class FinancialDataset(ABC):
    """Base class for a loaded, in-memory financial dataset.

    Subclasses wrap a specific source (synthetic generator, cached parquet,
    a market-data provider) but must expose the same walk-forward-safe
    splitting API so no algorithm accidentally trains on future data.
    """

    @property
    @abstractmethod
    def descriptor(self) -> DatasetDescriptor: ...

    @abstractmethod
    def features(self) -> np.ndarray:
        """Shape (n_samples, n_features)."""

    @abstractmethod
    def targets(self) -> np.ndarray:
        """Shape (n_samples,) or (n_samples, n_targets)."""

    def split_walk_forward(
        self, n_splits: int, min_train_fraction: float = 0.5
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) index pairs with strictly increasing,
        non-overlapping test windows and no test index ever preceding a
        train index — the minimum bar for not leaking future information
        into a financial time-series model.
        """
        n = len(self.targets())
        min_train = int(n * min_train_fraction)
        remaining = n - min_train
        if n_splits < 1 or remaining < n_splits:
            raise ValueError(
                f"Cannot make {n_splits} walk-forward splits from {n} samples "
                f"with min_train_fraction={min_train_fraction}"
            )
        fold_size = remaining // n_splits
        splits = []
        for i in range(n_splits):
            train_end = min_train + i * fold_size
            test_end = min_train + (i + 1) * fold_size if i < n_splits - 1 else n
            splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
        return splits
