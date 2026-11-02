"""Concrete dataset registry — an extension point over
core.dataset_interfaces.FinancialDataset, not a duplicate abstraction.

core.dataset_interfaces defines the FinancialDataset contract (features(),
targets(), split_walk_forward()). This module is where concrete, runnable
dataset *implementations* register themselves by name so experiment configs
can reference a dataset by string (e.g. "synthetic_garch_vol") instead of
importing a class directly.
"""
from __future__ import annotations

from core.dataset_interfaces import FinancialDataset
from core.registry import Registry

DATASET_REGISTRY: Registry = Registry("dataset")


def get_dataset(name: str, **kwargs) -> FinancialDataset:
    factory = DATASET_REGISTRY.get(name)
    return factory(**kwargs)
