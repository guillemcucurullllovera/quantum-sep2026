import numpy as np
import pytest

from benchmarks import MethodOutput, check_reproducibility, render_comparison_chart, run_benchmark
from core.errors import ValidationError
from core.registry import Registry


def test_run_benchmark_captures_both_methods():
    result = run_benchmark(
        method_a=lambda: MethodOutput(1.0, circuit_depth=12, two_qubit_gate_count=8, shots=1024),
        method_b=lambda: MethodOutput(1.05),
        method_a_name="quantum_method",
        method_b_name="classical_baseline",
        metric_name="objective",
        seed=42,
    )
    assert result.value_a == 1.0
    assert result.value_b == 1.05
    assert result.circuit_depth == 12
    assert result.wall_time_a_s is not None and result.wall_time_a_s >= 0
    assert result.seed == 42


def test_render_comparison_chart_writes_a_real_file(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_SEP2026_RESULTS_DIR", str(tmp_path))
    from core.config import get_settings

    get_settings(refresh=True)

    result = run_benchmark(
        method_a=lambda: MethodOutput(1.0),
        method_b=lambda: MethodOutput(1.1),
        method_a_name="a",
        method_b_name="b",
        metric_name="objective",
    )
    path = render_comparison_chart([result], run_id="test_run_123")
    import os

    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_reproducibility_check_passes_for_seeded_deterministic_function():
    def deterministic_fn() -> float:
        rng = np.random.default_rng(123)
        return float(rng.normal())

    check = check_reproducibility(deterministic_fn, seed=1)
    assert check.is_reproducible


def test_reproducibility_check_catches_a_genuinely_nondeterministic_function():
    # deliberately ignores the seed argument entirely -> must be flagged
    counter = {"n": 0}

    def nondeterministic_fn() -> float:
        counter["n"] += 1
        return float(counter["n"])  # different every call regardless of seeding

    check = check_reproducibility(nondeterministic_fn, seed=1)
    assert not check.is_reproducible
    assert check.max_abs_diff > 0


def test_generic_runner_raises_clear_error_for_unregistered_algorithm():
    from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig
    from experiments.runners import run_experiment

    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=4)
    config = ExperimentConfig(name="t", algorithm="definitely_not_registered_xyz", backend=backend)
    with pytest.raises(ValidationError, match="Unknown algorithm"):
        run_experiment(config)
