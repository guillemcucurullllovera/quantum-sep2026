"""CLI runner: `python -m applications.portfolio.cli --config <path.json>`

Loads a JSON config, runs one of three real pipelines (a single solve, a
multi-generation comparison, or a walk-forward backtest), and prints the
resulting report. Every mode calls the exact same
`solve_portfolio`/`compare_solvers`/`compare_walk_forward` functions this
application layer exposes everywhere else -- nothing here is CLI-specific
logic duplicating what the library already does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioSpec, SolverChoice
from applications.portfolio.multi_solver import ALL_SOLVER_CHOICES, compare_solvers, solve_portfolio
from applications.portfolio.report import build_comparison_report, build_report, build_walk_forward_report
from applications.portfolio.scenario import compare_walk_forward, generate_multi_period_market
from applications.portfolio.synthetic_market import generate_synthetic_market


def _build_spec_from_config(cfg: dict) -> PortfolioSpec:
    if "assets" in cfg:
        return PortfolioSpec.model_validate(cfg)

    n_assets = cfg["n_assets"]
    seed = cfg.get("seed", 0)
    mu, cov, names = generate_synthetic_market(n_assets, seed=seed)
    constraints = [ConstraintSpec(kind="cardinality", params={"k": cfg["cardinality_k"]})]
    for extra in cfg.get("constraints", []):
        constraints.append(ConstraintSpec(kind=extra["kind"], params=extra.get("params", {})))
    return PortfolioSpec(
        assets=[AssetSpec(name=nm, expected_return=float(m)) for nm, m in zip(names, mu)],
        cov_matrix=cov.tolist(), risk_aversion=cfg.get("risk_aversion", 0.5), constraints=constraints,
    )


def _run_single(cfg: dict) -> str:
    spec = _build_spec_from_config(cfg)
    choice = SolverChoice(cfg.get("solver", "qaoa_direct"))
    result = solve_portfolio(spec, choice, seed=cfg.get("seed", 42))
    return build_report(spec, result).summary


def _run_compare(cfg: dict) -> str:
    spec = _build_spec_from_config(cfg)
    requested = cfg.get("solvers")
    choices = [SolverChoice(c) for c in requested] if requested else list(ALL_SOLVER_CHOICES)
    comparison = compare_solvers(spec, seed=cfg.get("seed", 42), choices=choices)
    return build_comparison_report(spec, comparison)


def _run_walk_forward(cfg: dict) -> str:
    n_assets = cfg["n_assets"]
    n_periods = cfg.get("n_periods", 40)
    seed = cfg.get("seed", 0)
    realized, names, _sectors = generate_multi_period_market(
        n_assets, n_periods, seed=seed, n_sectors=cfg.get("n_sectors", 3)
    )
    requested = cfg.get("solvers")
    choices = (
        [SolverChoice(c) for c in requested]
        if requested
        else [SolverChoice.QAOA_DIRECT, SolverChoice.QOKIT_FAST, SolverChoice.CLASSICAL_BASELINE]
    )
    results = compare_walk_forward(
        realized, names, cardinality_k=cfg["cardinality_k"], solvers=choices,
        n_splits=cfg.get("n_splits", 4), turnover_limit=cfg.get("turnover_limit"), seed=seed,
    )
    return build_walk_forward_report(results)


_MODES = {"single": _run_single, "compare": _run_compare, "walk_forward": _run_walk_forward}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m applications.portfolio.cli",
        description="Run a portfolio optimization pipeline (single solve, multi-generation "
        "comparison, or walk-forward backtest) from a JSON config and print the report.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a JSON config file")
    parser.add_argument(
        "--mode", choices=sorted(_MODES), default=None,
        help="Overrides config['mode'] if given (default: config['mode'], else 'single')",
    )
    args = parser.parse_args(argv)

    cfg = json.loads(args.config.read_text())
    mode = args.mode or cfg.get("mode", "single")
    if mode not in _MODES:
        parser.error(f"unknown mode {mode!r}, expected one of {sorted(_MODES)}")

    print(_MODES[mode](cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
