"""Command-line runner for the tail-risk application: multi-method
VaR/CVaR comparison, VaR-breach backtesting, and walk-forward evaluation
over a synthetic (default) or CSV-supplied P&L series.

    python -m applications.tail_risk.cli
    python -m applications.tail_risk.cli --alpha 0.99 --backtest-splits 5
    python -m applications.tail_risk.cli --csv pnl.csv --column pnl --no-quantum
    python -m applications.tail_risk.cli --log-registry
"""
from __future__ import annotations

import argparse
import csv as csv_mod

import numpy as np

from applications.tail_risk.comparison import compare_methods
from applications.tail_risk.domain import TailRiskSpec
from applications.tail_risk.integration import log_comparison
from applications.tail_risk.walk_forward import rolling_tail_risk


def _load_pnl(args: argparse.Namespace) -> np.ndarray:
    if args.csv:
        with open(args.csv, newline="") as f:
            reader = csv_mod.DictReader(f)
            values = [float(row[args.column]) for row in reader]
        return np.asarray(values, dtype=float)
    rng = np.random.default_rng(args.seed)
    # Student-t innovations: fatter tails than Gaussian noise -- a more
    # honest synthetic default for a tail-risk demo than plain N(0, 1).
    t = rng.standard_t(df=args.t_df, size=args.n_samples)
    return t * args.scale + args.drift


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tail_risk")
    parser.add_argument("--csv", type=str, default=None, help="CSV file with a P&L column (default: synthetic data)")
    parser.add_argument("--column", type=str, default="pnl")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--t-df", type=float, default=4.0, help="Student-t degrees of freedom for synthetic data")
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--drift", type=float, default=0.0002)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-quantum", action="store_true")
    parser.add_argument("--n-grid-qubits", type=int, default=3)
    parser.add_argument(
        "--backtest-splits", type=int, default=0, help="If > 0, also run an N-fold walk-forward backtest",
    )
    parser.add_argument("--log-registry", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pnl = _load_pnl(args)

    spec = TailRiskSpec(
        confidence_levels=[args.alpha], seed=args.seed,
        run_quantum=not args.no_quantum, n_grid_qubits=args.n_grid_qubits,
    )
    report = compare_methods(pnl, spec)

    print(f"Tail-risk comparison: n={report.n_samples} alpha={report.alpha} prob_of_loss={report.prob_of_loss:.4f}")
    print(f"{'method':<28}{'VaR':>12}{'CVaR':>12}{'wall_time_s':>14}")
    for name, m in report.results.items():
        print(f"{name:<28}{m.var:>12.5f}{m.cvar:>12.5f}{m.wall_time_s:>14.4f}")

    if report.notes:
        print("\nDiagnostics:")
        for note in report.notes:
            print(f"  - {note}")

    if args.backtest_splits > 0:
        wf = rolling_tail_risk(pnl, n_splits=args.backtest_splits, alpha=args.alpha)
        print(f"\nWalk-forward ({wf.n_splits} folds):")
        for fold in wf.folds:
            bt = fold.breach_test
            print(
                f"  fold {fold.fold}: train={fold.n_train} test={fold.n_test} "
                f"VaR={fold.var_estimate:.5f} CVaR={fold.cvar_estimate:.5f} "
                f"breaches={bt.n_breaches}/{bt.n_obs} (expected {bt.expected_rate:.3f}) "
                f"kupiec_p={bt.kupiec_p_value:.4f}"
            )

    if args.log_registry:
        run_id = log_comparison(report, seed=args.seed)
        print(f"\nLogged to experiment registry: run_id={run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
