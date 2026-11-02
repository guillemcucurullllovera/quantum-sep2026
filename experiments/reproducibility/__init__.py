"""Full-pipeline reproducibility: re-run an entire ExperimentConfig (not just
a bare function, as benchmarks.reproducibility already checks) twice with
the same seed through experiments.runners.run_experiment, and confirm every
numeric metric matches within tolerance.

This is a different, stronger guarantee than benchmarks.reproducibility's
single-callable check: it exercises the real config-loading, registry
dispatch, and result-logging path end to end, which is where non-determinism
actually tends to creep in (e.g. an algorithm reading global RNG state before
core.seeds.seed_everything runs, or a dataset factory that isn't itself
seeded).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.experiment_registry import ExperimentRegistry
from core.schemas import ExperimentConfig, ExperimentResult
from experiments.runners.generic_runner import run_experiment


@dataclass
class ConfigReproducibilityReport:
    config_name: str
    is_reproducible: bool
    mismatched_metrics: dict[str, tuple[float, float]] = field(default_factory=dict)
    run_id_a: str = ""
    run_id_b: str = ""


def check_config_reproducibility(
    config: ExperimentConfig, *, tolerance: float = 1e-9, registry: ExperimentRegistry | None = None
) -> ConfigReproducibilityReport:
    registry = registry or ExperimentRegistry()

    result_a: ExperimentResult = run_experiment(config, registry=registry)
    result_b: ExperimentResult = run_experiment(config, registry=registry)

    mismatched: dict[str, tuple[float, float]] = {}
    all_keys = set(result_a.metrics) | set(result_b.metrics)
    for k in all_keys:
        va, vb = result_a.metrics.get(k), result_b.metrics.get(k)
        if va is None or vb is None or abs(va - vb) > tolerance:
            mismatched[k] = (va, vb)

    return ConfigReproducibilityReport(
        config_name=config.name,
        is_reproducible=len(mismatched) == 0,
        mismatched_metrics=mismatched,
        run_id_a=result_a.metadata.run_id,
        run_id_b=result_b.metadata.run_id,
    )


def check_many(configs: list[ExperimentConfig], **kwargs) -> list[ConfigReproducibilityReport]:
    return [check_config_reproducibility(c, **kwargs) for c in configs]
