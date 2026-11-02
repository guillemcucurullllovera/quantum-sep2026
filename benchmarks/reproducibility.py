"""Reproducibility harness: run the same seeded callable twice and confirm
identical results. A real regression-safety mechanism — every algorithm
module should be checked with this before being trusted for benchmarking,
since a "benchmark" against a nondeterministic method is meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.seeds import seed_everything


@dataclass
class ReproducibilityCheckResult:
    is_reproducible: bool
    first_value: float
    second_value: float
    max_abs_diff: float


def check_reproducibility(
    fn: Callable[[], float], *, seed: int, tolerance: float = 1e-9
) -> ReproducibilityCheckResult:
    """`fn` should be a zero-arg callable that internally uses whatever
    randomness it needs; we reseed identically before each call so the
    comparison isolates whether `fn` itself is deterministic given a seed.
    """
    seed_everything(seed)
    first = float(fn())

    seed_everything(seed)
    second = float(fn())

    diff = abs(first - second)
    return ReproducibilityCheckResult(
        is_reproducible=diff <= tolerance,
        first_value=first,
        second_value=second,
        max_abs_diff=diff,
    )
