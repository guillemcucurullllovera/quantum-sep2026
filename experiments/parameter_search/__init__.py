"""Generic parameter-search harness over any registered algorithm.

Builds a grid or random sample of parameter dicts, runs each through
experiments.runners.run_experiment, and returns the results — every run is
logged to the experiment registry like any other, so a sweep is just many
ordinary runs, not a special code path.
"""
from __future__ import annotations

import itertools
import random
from typing import Any, Callable, Iterable

from core.experiment_registry import ExperimentRegistry
from core.schemas import ExperimentConfig, ExperimentResult
from experiments.runners.generic_runner import run_experiment


def grid(param_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a {param_name: [values]} space."""
    if not param_space:
        return [{}]
    keys = list(param_space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(param_space[k] for k in keys))]


def random_sample(param_space: dict[str, list[Any]], n: int, *, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(param_space)
    return [{k: rng.choice(param_space[k]) for k in keys} for _ in range(n)]


def run_sweep(
    base_config: ExperimentConfig,
    param_sets: Iterable[dict[str, Any]],
    *,
    registry: ExperimentRegistry | None = None,
    on_error: str = "collect",
) -> list[ExperimentResult | Exception]:
    """Run base_config once per param_set, overriding config.params each time.

    on_error='collect' (default) puts the exception in the results list
    instead of aborting the whole sweep — a bad parameter combination
    shouldn't lose every other result. on_error='raise' stops immediately.
    """
    registry = registry or ExperimentRegistry()
    results: list[ExperimentResult | Exception] = []
    for params in param_sets:
        cfg = base_config.model_copy(update={"params": {**base_config.params, **params}})
        try:
            results.append(run_experiment(cfg, registry=registry))
        except Exception as e:  # noqa: BLE001 - intentionally broad, see on_error
            if on_error == "raise":
                raise
            results.append(e)
    return results
