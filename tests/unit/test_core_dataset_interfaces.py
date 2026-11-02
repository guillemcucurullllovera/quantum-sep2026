import numpy as np
import pytest

from core.dataset_interfaces import FinancialDataset
from core.schemas import DatasetDescriptor


class _ToyDataset(FinancialDataset):
    def __init__(self, n=100):
        self._X = np.arange(n).reshape(-1, 1).astype(float)
        self._y = np.arange(n).astype(float)

    @property
    def descriptor(self):
        return DatasetDescriptor(name="toy", n_samples=len(self._y), n_features=1, domain="test")

    def features(self):
        return self._X

    def targets(self):
        return self._y


def test_walk_forward_splits_never_leak_future():
    ds = _ToyDataset(100)
    splits = ds.split_walk_forward(n_splits=4, min_train_fraction=0.5)
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()  # no leakage: train strictly precedes test


def test_walk_forward_rejects_too_many_splits():
    ds = _ToyDataset(10)
    with pytest.raises(ValueError):
        ds.split_walk_forward(n_splits=100)
