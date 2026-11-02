import json

from core.schemas import ExperimentConfig
from experiments.configs import load_experiment_config, save_experiment_config


def test_save_and_load_round_trip(tmp_path):
    backend = {"name": "aer_simulator", "kind": "simulator", "num_qubits": 8}
    config = ExperimentConfig.model_validate(
        {"name": "t", "algorithm": "qaoa_qubo", "backend": backend, "params": {"n_assets": 6}}
    )
    path = tmp_path / "cfg.json"
    save_experiment_config(config, path)

    loaded = load_experiment_config(path)
    assert loaded.name == "t"
    assert loaded.params["n_assets"] == 6


def test_load_missing_file_raises():
    try:
        load_experiment_config("/nonexistent/path/cfg.json")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
