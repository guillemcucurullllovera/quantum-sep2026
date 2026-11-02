"""CLI runner for the unified classification/regime/anomaly dispatcher.

    python -m applications.classification.cli list-algorithms
    python -m applications.classification.cli run --task-type binary --algorithm vqc_v5_reupload
    python -m applications.classification.cli compare --task-type binary
    python -m applications.classification.cli rolling --task-type binary --algorithm pqk_gen2 --n-splits 4

All three subcommands operate on the platform's synthetic datasets by
default (no market-data wiring required to exercise the dispatcher), and
print a compact human-readable summary -- this is the "does the whole
product layer actually work end to end" smoke path, complementary to
`core.cli` (which only lists registries).
"""
from __future__ import annotations

import argparse
import json
import sys

from applications.classification.benchmark_report import run_comprehensive_comparison
from applications.classification.dispatcher import dispatch, supported_task_types
from applications.classification.rolling_evaluation import run_rolling_evaluation
from applications.classification.task_spec import Algorithm, ClassificationTaskSpec, TaskType


def _default_dataset(task_type: TaskType):
    if task_type == TaskType.REGRESSION:
        from applications.classification.synthetic_volatility_dataset import SyntheticVolatilityDataset

        return SyntheticVolatilityDataset(n_samples=200, n_features=6, seed=13)
    if task_type in (TaskType.MULTICLASS, TaskType.REGIME):
        from applications.regime_detection.synthetic_regime_dataset import SyntheticRegimeDataset

        return SyntheticRegimeDataset(n_per_regime=80, n_features=6, seed=11)
    if task_type == TaskType.ANOMALY:
        from applications.anomaly_detection.synthetic_transactions import SyntheticTransactionDataset

        return SyntheticTransactionDataset(n_normal=160, n_anomaly=40, seed=0)
    from applications.classification.synthetic_financial_dataset import SyntheticDirectionalDataset

    return SyntheticDirectionalDataset(n_samples=200, n_features=6, seed=7)


def _split(dataset, test_fraction: float = 0.3):
    X, y = dataset.features(), dataset.targets()
    n_test = max(1, int(len(y) * test_fraction))
    n_train = len(y) - n_test
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _cmd_list_algorithms(args: argparse.Namespace) -> int:
    for algo in Algorithm:
        tasks = ", ".join(sorted(t.value for t in supported_task_types(algo)))
        print(f"{algo.value:32s} {tasks}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    task_type = TaskType(args.task_type)
    algorithm = Algorithm(args.algorithm)
    dataset = _default_dataset(task_type)
    X_train, y_train, X_test, y_test = _split(dataset, args.test_fraction)
    spec = ClassificationTaskSpec(
        task_type=task_type, algorithm=algorithm, n_qubits=args.n_qubits, seed=args.seed, calibrate=args.calibrate
    )
    result = dispatch(spec, X_train, y_train, X_test, y_test)
    print(json.dumps(result.to_summary_dict(), indent=2, default=str))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    task_type = TaskType(args.task_type)
    dataset = _default_dataset(task_type)
    X_train, y_train, X_test, y_test = _split(dataset, args.test_fraction)
    report = run_comprehensive_comparison(
        task_type, X_train, y_train, X_test, y_test, n_qubits=args.n_qubits, seed=args.seed
    )
    print(json.dumps(report.to_summary_dict(), indent=2, default=str))
    return 0


def _cmd_rolling(args: argparse.Namespace) -> int:
    task_type = TaskType(args.task_type)
    algorithm = Algorithm(args.algorithm)
    dataset = _default_dataset(task_type)
    spec = ClassificationTaskSpec(
        task_type=task_type, algorithm=algorithm, n_qubits=args.n_qubits, seed=args.seed, n_splits=args.n_splits
    )
    report = run_rolling_evaluation(spec, dataset)
    print(json.dumps(report.to_summary_dict(), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="applications.classification.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-algorithms").set_defaults(func=_cmd_list_algorithms)

    run_p = sub.add_parser("run")
    run_p.add_argument("--task-type", required=True, choices=[t.value for t in TaskType])
    run_p.add_argument("--algorithm", required=True, choices=[a.value for a in Algorithm])
    run_p.add_argument("--n-qubits", type=int, default=4)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--test-fraction", type=float, default=0.3)
    run_p.add_argument("--calibrate", action="store_true")
    run_p.set_defaults(func=_cmd_run)

    compare_p = sub.add_parser("compare")
    compare_p.add_argument("--task-type", required=True, choices=[t.value for t in TaskType])
    compare_p.add_argument("--n-qubits", type=int, default=4)
    compare_p.add_argument("--seed", type=int, default=42)
    compare_p.add_argument("--test-fraction", type=float, default=0.3)
    compare_p.set_defaults(func=_cmd_compare)

    rolling_p = sub.add_parser("rolling")
    rolling_p.add_argument("--task-type", required=True, choices=[t.value for t in TaskType])
    rolling_p.add_argument("--algorithm", required=True, choices=[a.value for a in Algorithm])
    rolling_p.add_argument("--n-qubits", type=int, default=4)
    rolling_p.add_argument("--seed", type=int, default=42)
    rolling_p.add_argument("--n-splits", type=int, default=3)
    rolling_p.set_defaults(func=_cmd_rolling)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
