import numpy as np
import pandas as pd
import pytest

from core.errors import ValidationError
from core.registry import ALGORITHM_REGISTRY
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata
from experiments.preprocessing import Pipeline
from experiments.reproducibility import check_config_reproducibility


def test_preprocessing_pipeline_zscore_step_runs():
    series = pd.Series(np.sin(np.linspace(0, 20, 200)) + np.random.default_rng(0).normal(0, 0.01, 200))
    pipeline = Pipeline.from_spec([{"name": "zscore", "params": {"window": 20}}])
    out = pipeline.run(series)
    assert len(out) > 0
    assert abs(out.mean()) < 1.0  # z-scored series should be roughly centered


def test_preprocessing_pipeline_rejects_unknown_step():
    with pytest.raises(ValueError):
        Pipeline.from_spec([{"name": "not_a_real_step"}])


@ALGORITHM_REGISTRY.register("_wave3_test_conformant_algo")
def _conformant_test_algorithm(config: ExperimentConfig, seed: int) -> ExperimentResult:
    """A correctly-conformant algorithm (config, seed) -> ExperimentResult,
    used to prove experiments.reproducibility's harness works correctly
    against the contract experiments.runners.generic_runner actually expects
    -- see benchmarks/quantum_vs_classical/qaoa.py's module docstring for
    why the *real* registered algorithms (qaoa_qubo, qrc, pqk_gen1/v2) do
    NOT conform to this contract and can't be used for this specific test.
    """
    value = float(config.params.get("x", 0)) ** 2 + seed * 0  # seed included in signature per contract, unused here
    return ExperimentResult(
        metadata=RunMetadata(algorithm="_wave3_test_conformant_algo"),
        config=config,
        metrics={"objective": value},
    )


def test_config_reproducibility_harness_detects_deterministic_pipeline():
    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=4)
    config = ExperimentConfig(
        name="repro_test", algorithm="_wave3_test_conformant_algo", backend=backend, params={"x": 3.0}
    )
    report = check_config_reproducibility(config)
    assert report.is_reproducible
    assert report.mismatched_metrics == {}


def test_config_reproducibility_harness_catches_nondeterminism():
    import random

    @ALGORITHM_REGISTRY.register("_wave3_test_nondeterministic_algo")
    def _nondeterministic(config: ExperimentConfig, seed: int) -> ExperimentResult:
        # deliberately ignores the seed to prove the harness catches this
        return ExperimentResult(
            metadata=RunMetadata(algorithm="_wave3_test_nondeterministic_algo"),
            config=config,
            metrics={"objective": random.SystemRandom().random()},
        )

    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=4)
    config = ExperimentConfig(name="repro_negative_test", algorithm="_wave3_test_nondeterministic_algo", backend=backend)
    report = check_config_reproducibility(config)
    assert not report.is_reproducible
    assert "objective" in report.mismatched_metrics
