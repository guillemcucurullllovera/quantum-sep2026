"""End-to-end coverage for `python -m applications.portfolio.cli` (all three
modes: single solve, multi-generation comparison, walk-forward backtest).
"""
from __future__ import annotations

import json

from applications.portfolio.cli import main


def test_cli_single_mode_end_to_end(tmp_path, capsys):
    cfg = {
        "mode": "single", "n_assets": 6, "cardinality_k": 3, "seed": 1,
        "solver": "classical_baseline",
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    rc = main(["--config", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "classical_baseline" in out
    assert "cost=" in out


def test_cli_compare_mode_with_explicit_solver_subset(tmp_path, capsys):
    cfg = {
        "mode": "compare", "n_assets": 6, "cardinality_k": 3, "seed": 2,
        "solvers": ["qokit_fast", "classical_baseline", "modeled_repair"],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    rc = main(["--config", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "qokit_fast" in out and "classical_baseline" in out and "modeled_repair" in out


def test_cli_mode_flag_overrides_config_mode(tmp_path, capsys):
    cfg = {"mode": "single", "n_assets": 5, "cardinality_k": 2, "seed": 3, "solver": "qokit_fast"}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    rc = main(["--config", str(cfg_path), "--mode", "single"])
    assert rc == 0


def test_cli_walk_forward_mode_end_to_end(tmp_path, capsys):
    cfg = {
        "mode": "walk_forward", "n_assets": 6, "n_periods": 24, "cardinality_k": 3,
        "n_splits": 3, "turnover_limit": 3, "seed": 4,
        "solvers": ["classical_baseline", "qokit_fast"],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    rc = main(["--config", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Walk-forward comparison" in out
    assert "classical_baseline" in out and "qokit_fast" in out


def test_cli_rejects_unknown_mode(tmp_path):
    cfg = {"mode": "not_a_real_mode", "n_assets": 4, "cardinality_k": 2}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    try:
        main(["--config", str(cfg_path)])
        assert False, "expected SystemExit from argparse.error"
    except SystemExit as e:
        assert e.code != 0


def test_cli_accepts_full_portfolio_spec_config(tmp_path, capsys):
    cfg = {
        "mode": "single",
        "assets": [{"name": "A", "expected_return": 0.1}, {"name": "B", "expected_return": 0.05}, {"name": "C", "expected_return": 0.08}],
        "cov_matrix": [[0.02, 0.001, 0.0], [0.001, 0.015, 0.0], [0.0, 0.0, 0.01]],
        "constraints": [{"kind": "cardinality", "params": {"k": 2}}],
        "solver": "classical_baseline",
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    rc = main(["--config", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "selected 2/3 assets" in out
