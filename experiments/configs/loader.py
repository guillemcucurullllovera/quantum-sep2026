"""Experiment config format: plain JSON matching core.schemas.ExperimentConfig
field-for-field. JSON (not YAML) deliberately — no new dependency, and
core.serialization already round-trips this shape safely.

Example file:
{
  "name": "qaoa_portfolio_smoke_test",
  "algorithm": "qaoa_qubo",
  "application": "portfolio",
  "backend": {"name": "aer_simulator", "kind": "simulator", "num_qubits": 8},
  "shots": 4096,
  "seed": 42,
  "params": {"n_assets": 6, "cardinality": 3}
}
"""
from __future__ import annotations

from pathlib import Path

from core.schemas import ExperimentConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No experiment config at {path}")
    return ExperimentConfig.model_validate_json(path.read_text())


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2))
