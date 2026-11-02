"""Generic experiment runner: looks up an algorithm by name in
ALGORITHM_REGISTRY, runs it against the given ExperimentConfig, and logs the
result to the experiment registry.

An "algorithm" registered here is expected to be a callable (or a class with
a `run(config: ExperimentConfig, *, seed: int) -> ExperimentResult` method)
— see core.registry for the registration decorator each algorithm module
uses. This runner intentionally does not import any specific algorithm
module: it discovers them purely through the registry so new algorithms
never require editing this file.
"""
from __future__ import annotations

from core.errors import ValidationError
from core.experiment_registry import ExperimentRegistry
from core.registry import ALGORITHM_REGISTRY
from core.schemas import ExperimentConfig, ExperimentResult
from core.seeds import seed_everything


def run_experiment(
    config: ExperimentConfig, *, registry: ExperimentRegistry | None = None
) -> ExperimentResult:
    algo = ALGORITHM_REGISTRY.get(config.algorithm)  # raises ValidationError if unregistered

    seed_everything(config.seed)

    if hasattr(algo, "run"):
        result = algo.run(config, seed=config.seed)
    elif callable(algo):
        result = algo(config, seed=config.seed)
    else:
        raise ValidationError(
            f"Registered algorithm '{config.algorithm}' is neither callable nor "
            f"has a .run(config, seed) method — cannot execute it."
        )

    if not isinstance(result, ExperimentResult):
        raise ValidationError(
            f"Algorithm '{config.algorithm}' must return an ExperimentResult, "
            f"got {type(result)!r}"
        )

    reg = registry or ExperimentRegistry()
    reg.record(result)
    return result
