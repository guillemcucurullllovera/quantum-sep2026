"""End-to-end example: build an ExperimentConfig, save it to disk, and run
it through tools.experiment_launch exactly the way a user would from the
command line. Run:
    python -m examples.experiment_config_driven_example
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from core.registry import ALGORITHM_REGISTRY
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata
from experiments.configs.loader import save_experiment_config
from tools.experiment_launch import main as launch_main


@ALGORITHM_REGISTRY.register("examples_demo_algorithm")
def _demo_algorithm(config: ExperimentConfig, *, seed: int) -> ExperimentResult:
    """A minimal but properly-conformant example algorithm: takes a config
    and seed, returns an ExperimentResult -- the contract every registered
    algorithm should follow (see experiments/runners/generic_runner.py)."""
    meta = RunMetadata(algorithm=config.algorithm, generation="example")
    return ExperimentResult(metadata=meta, config=config, metrics={"demo_metric": float(seed) ** 0.5})


def main() -> None:
    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=2)
    config = ExperimentConfig(name="config_driven_demo", algorithm="examples_demo_algorithm", backend=backend, seed=9)

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "demo_config.json"
        save_experiment_config(config, config_path)
        print(f"Saved config to {config_path}, launching via tools.experiment_launch...")
        launch_main(["--config", str(config_path)])


if __name__ == "__main__":
    main()
