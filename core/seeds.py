"""Reproducibility helpers.

Every ExperimentConfig carries a seed; algorithms MUST call seed_everything
at the start of a run rather than relying on ambient global state, so that
two runs with the same config+seed produce comparable results.
"""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        from qiskit.utils import algorithm_globals  # type: ignore

        algorithm_globals.random_seed = seed
    except ImportError:
        pass
