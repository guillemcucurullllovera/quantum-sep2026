"""CLI runner for the options/derivatives/hedging application layer.

    python -m applications.options.cli price --payoff call_spread --spot 100 \
        --strike 95 --strike2 115 --method qae_mlae
    python -m applications.options.cli compare --payoff up_and_out_call \
        --spot 100 --strike 100 --barrier 115
    python -m applications.options.cli resources --payoff up_and_out_call \
        --spot 100 --strike 100 --barrier 115 --target-epsilon 0.01
    python -m applications.options.cli convergence --payoff up_and_out_call \
        --spot 100 --strike 100 --barrier 115
    python -m applications.options.cli greeks --payoff european_call --spot 100 --strike 100
    python -m applications.options.cli hedge --notional 1000000 --spot 100 \
        --realized-vol 0.3 --regime trend_down
"""
from __future__ import annotations

import argparse
import sys

from applications.options.comparison_report import build_comparison_report
from applications.options.domain import OptionSpec, PayoffKind
from applications.options.greeks import finite_difference_greeks
from applications.options.pricing_methods import PricingMethod, price_option
from applications.options.resource_and_convergence import estimate_resources, run_convergence_study
from core.errors import QuantumSep2026Error


def _add_spec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--payoff", required=True, choices=[k.value for k in PayoffKind])
    p.add_argument("--spot", type=float, required=True)
    p.add_argument("--strike", type=float, required=True)
    p.add_argument("--strike2", type=float, default=None)
    p.add_argument("--barrier", type=float, default=None)
    p.add_argument("--spot2", type=float, default=None)
    p.add_argument("--vol2", type=float, default=None)
    p.add_argument("--correlation", type=float, default=None)
    p.add_argument("--rate", type=float, default=0.04)
    p.add_argument("--vol", type=float, default=0.20)
    p.add_argument("--maturity", type=float, default=0.25)


def _spec_from_args(args: argparse.Namespace) -> OptionSpec:
    return OptionSpec(
        payoff=PayoffKind(args.payoff), spot=args.spot, strike=args.strike, strike2=args.strike2,
        barrier=args.barrier, spot2=args.spot2, vol2=args.vol2, correlation=args.correlation,
        rate=args.rate, vol=args.vol, maturity=args.maturity,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="applications.options.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    price_p = sub.add_parser("price")
    _add_spec_args(price_p)
    price_p.add_argument("--method", default="auto", choices=[m.value for m in PricingMethod])
    price_p.add_argument("--shots", type=int, default=4096)
    price_p.add_argument("--n-qubits", type=int, default=None)

    compare_p = sub.add_parser("compare")
    _add_spec_args(compare_p)
    compare_p.add_argument("--shots", type=int, default=4096)

    resources_p = sub.add_parser("resources")
    _add_spec_args(resources_p)
    resources_p.add_argument("--target-epsilon", type=float, default=0.01)
    resources_p.add_argument("--n-qubits", type=int, default=None)

    convergence_p = sub.add_parser("convergence")
    _add_spec_args(convergence_p)

    greeks_p = sub.add_parser("greeks")
    _add_spec_args(greeks_p)
    greeks_p.add_argument("--method", default="auto", choices=[m.value for m in PricingMethod])

    hedge_p = sub.add_parser("hedge")
    hedge_p.add_argument("--notional", type=float, required=True)
    hedge_p.add_argument("--spot", type=float, required=True)
    hedge_p.add_argument("--realized-vol", type=float, required=True)
    hedge_p.add_argument("--forecast-vol", type=float, default=None)
    hedge_p.add_argument("--rate", type=float, default=0.04)
    hedge_p.add_argument("--horizon", type=float, default=0.25)
    hedge_p.add_argument("--regime", default="trend_down", choices=["trend_up", "trend_down", "ranging"])

    args = parser.parse_args(argv)

    try:
        if args.command == "price":
            spec = _spec_from_args(args)
            result = price_option(spec, PricingMethod(args.method), shots=args.shots, n_qubits=args.n_qubits)
            print(f"price:                {result.price:.4f}")
            print(f"method:               {result.method.value}")
            print(f"classical reference:  {result.classical_reference_price} ({result.reference_method})")
            print(f"abs/rel error:        {result.abs_error} / {result.rel_error}")
            print(f"note:                 {result.note}")

        elif args.command == "compare":
            spec = _spec_from_args(args)
            report = build_comparison_report(spec, shots=args.shots)
            for row in report.as_table():
                print(row)

        elif args.command == "resources":
            spec = _spec_from_args(args)
            estimate = estimate_resources(spec, target_epsilon=args.target_epsilon, n_qubits=args.n_qubits)
            print(f"qubits:                {estimate.circuit_stats.n_qubits if estimate.circuit_stats else 'n/a'}")
            print(f"depth:                 {estimate.circuit_stats.depth if estimate.circuit_stats else 'n/a'}")
            print(f"canonical AE queries:  {estimate.canonical_ae_queries}")
            print(f"IQAE expected queries: {estimate.iqae_expected_queries}")
            print(f"note:                  {estimate.note}")

        elif args.command == "convergence":
            spec = _spec_from_args(args)
            study = run_convergence_study(spec)
            for row in study.as_table():
                print(row)

        elif args.command == "greeks":
            spec = _spec_from_args(args)
            g = finite_difference_greeks(spec, method=PricingMethod(args.method))
            print(f"delta: {g.delta:.4f}  (analytic: {g.analytic_delta})")
            print(f"vega:  {g.vega:.4f}  (analytic: {g.analytic_vega})")

        elif args.command == "hedge":
            from applications.hedging.options_based_hedge import (
                PortfolioExposure, compute_hedge_ratio, simulate_hedge_effectiveness,
            )

            exposure = PortfolioExposure(
                notional=args.notional, spot=args.spot, realized_vol=args.realized_vol,
                rate=args.rate, horizon=args.horizon,
            )
            hedge = compute_hedge_ratio(exposure, regime=args.regime, forecast_vol=args.forecast_vol)
            report = simulate_hedge_effectiveness(exposure, hedge)
            print(f"regime:               {args.regime}  coverage={hedge.coverage}")
            print(f"put strike/premium:   {hedge.put_strike:.2f} / {hedge.put_premium:.4f}")
            print(f"unhedged CVaR:        {report.unhedged_cvar:.2f}")
            print(f"hedged CVaR:          {report.hedged_cvar:.2f}")
            print(f"CVaR reduction:       {report.cvar_reduction:.2f} ({report.cvar_reduction_fraction:.1%})")
            print(f"hedge cost/notional:  {report.hedge_cost_as_fraction_of_notional:.4%}")
        else:  # pragma: no cover - argparse enforces valid subcommands
            return 1
    except QuantumSep2026Error as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
