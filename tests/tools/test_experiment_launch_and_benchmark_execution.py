"""Tests for tools.experiment_launch and tools.benchmark_execution.

Uses a locally-registered, properly-conformant test algorithm (matching the
(config, seed) -> ExperimentResult contract) rather than any of the
platform's real algorithms, since a known pre-existing gap (documented by
the experiments/benchmarks workstream) means several registered
algorithms don't uniformly conform to that contract yet. This proves the
CLI tools themselves are correct, independent of that separate gap.
"""
import json

from core.registry import ALGORITHM_REGISTRY
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata
from core.experiment_registry import ExperimentRegistry
from experiments.configs.loader import save_experiment_config
from tools.experiment_launch import main as launch_main
from tools.benchmark_execution import main as bench_main, register_quick_benchmark, _QUICK_BENCHMARKS
from benchmarks.runner import MethodOutput


def _conformant_algo(config: ExperimentConfig, *, seed: int) -> ExperimentResult:
    meta = RunMetadata(algorithm=config.algorithm)
    return ExperimentResult(metadata=meta, config=config, metrics={"objective": float(seed)})


def test_experiment_launch_cli_runs_end_to_end(tmp_path, monkeypatch, capsys):
    algo_name = "_wave4_test_conformant_algo"
    if algo_name not in ALGORITHM_REGISTRY.list():
        ALGORITHM_REGISTRY.register(algo_name)(_conformant_algo)

    db = tmp_path / "reg.sqlite3"
    monkeypatch.setattr("experiments.runners.generic_runner.ExperimentRegistry", lambda: ExperimentRegistry(db_path=db))

    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=2)
    config = ExperimentConfig(name="launch_test", algorithm=algo_name, backend=backend, seed=7)
    config_path = tmp_path / "config.json"
    save_experiment_config(config, config_path)

    rc = launch_main(["--config", str(config_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Done. run_id=" in out
    assert "objective: 7" in out


def test_experiment_launch_reports_failure_for_unregistered_algorithm(tmp_path):
    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=2)
    config = ExperimentConfig(name="t", algorithm="totally_unregistered_xyz", backend=backend)
    config_path = tmp_path / "config.json"
    save_experiment_config(config, config_path)

    rc = launch_main(["--config", str(config_path)])
    assert rc == 1


def test_benchmark_execution_cli_compares_two_quick_benchmarks(capsys):
    register_quick_benchmark("_wave4_a", lambda: MethodOutput(1.0))
    register_quick_benchmark("_wave4_b", lambda: MethodOutput(2.0))

    rc = bench_main(["--method-a", "_wave4_a", "--method-b", "_wave4_b", "--metric-name", "value"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "_wave4_a: 1.0" in out
    assert "_wave4_b: 2.0" in out


def test_benchmark_execution_cli_unknown_method_fails_cleanly(capsys):
    rc = bench_main(["--method-a", "nope", "--method-b", "_wave4_a"])
    assert rc == 1
