"""CLI runner for the multi-method time-series / volatility forecast
comparison.

    python -m applications.time_series.cli
    python -m applications.time_series.cli --dataset regime_switching --n-steps 800 \\
        --methods esn qrc_gen2 mamba_ssm naive_constant --noise
    python -m applications.time_series.cli --dataset garch --fracdiff --n-splits 6
"""
from __future__ import annotations

import argparse

from applications.time_series.dispatcher import run_dispatcher
from applications.time_series.domain import (
    FeatureSpec,
    MethodChoice,
    NoiseSpec,
    QRCHyperparams,
    TargetKind,
    TimeSeriesForecastSpec,
    ValidationStrategy,
)
from applications.time_series.report import format_report
from applications.time_series.synthetic import RegimeSwitchingVolatilityDataset, SyntheticVolatilityDataset
from core.dataset_interfaces import FinancialDataset


def _build_dataset(name: str, n_steps: int, seed: int) -> FinancialDataset:
    if name == "regime_switching":
        return RegimeSwitchingVolatilityDataset(n_steps=n_steps, seed=seed)
    return SyntheticVolatilityDataset(n_steps=n_steps, seed=seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="applications.time_series.cli")
    parser.add_argument("--dataset", choices=["garch", "regime_switching"], default="regime_switching")
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--methods", nargs="*", choices=[m.value for m in MethodChoice], default=None)
    parser.add_argument("--validation", choices=["walk_forward", "single_split"], default="walk_forward")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--qrc-n1", type=int, default=5)
    parser.add_argument("--qrc-n2", type=int, default=5)
    parser.add_argument("--fracdiff", action="store_true", help="apply fracdiff preprocessing before forecasting")
    parser.add_argument("--noise", action="store_true", help="also run the noisy-vs-noiseless QRC diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-log", action="store_true", help="skip logging results into the experiment registry")
    args = parser.parse_args(argv)

    dataset = _build_dataset(args.dataset, args.n_steps, args.seed)
    methods = [MethodChoice(m) for m in args.methods] if args.methods else list(MethodChoice)

    spec = TimeSeriesForecastSpec(
        target_kind=TargetKind.REALIZED_VOLATILITY,
        horizon=args.horizon,
        methods=methods,
        validation=ValidationStrategy(args.validation),
        n_splits=args.n_splits,
        features=FeatureSpec(use_fracdiff=args.fracdiff),
        noise=NoiseSpec(enabled=args.noise),
        qrc=QRCHyperparams(n1=args.qrc_n1, n2=args.qrc_n2),
        seed=args.seed,
    )

    result = run_dispatcher(dataset, spec, log_experiments=not args.no_log)
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
