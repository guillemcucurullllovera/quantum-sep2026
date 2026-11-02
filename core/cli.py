"""Minimal operator CLI over the registries + experiment registry.

    python -m core.cli list-algorithms
    python -m core.cli list-backends
    python -m core.cli list-applications
    python -m core.cli history --algorithm qaoa_qubo
"""
from __future__ import annotations

import argparse
import sys

from core.experiment_registry import ExperimentRegistry
from core.registry import ALGORITHM_REGISTRY, APPLICATION_REGISTRY, BACKEND_REGISTRY


def _import_all_modules() -> None:
    """Import every algorithms/applications/hardware package so their
    @REGISTRY.register(...) decorators have run before we list them."""
    import importlib
    import pkgutil

    for top in ("algorithms", "algorithms_extended", "applications", "hardware"):
        try:
            pkg = importlib.import_module(top)
        except ImportError:
            continue
        if not hasattr(pkg, "__path__"):
            continue
        for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=f"{top}."):
            try:
                importlib.import_module(name)
            except Exception as e:  # pragma: no cover - CLI diagnostic path
                print(f"warning: failed to import {name}: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantum_sep2026")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-algorithms")
    sub.add_parser("list-backends")
    sub.add_parser("list-applications")
    history = sub.add_parser("history")
    history.add_argument("--algorithm", required=True)

    args = parser.parse_args(argv)
    _import_all_modules()

    if args.command == "list-algorithms":
        for name in ALGORITHM_REGISTRY.list():
            print(name)
    elif args.command == "list-backends":
        for name in BACKEND_REGISTRY.list():
            print(name)
    elif args.command == "list-applications":
        for name in APPLICATION_REGISTRY.list():
            print(name)
    elif args.command == "history":
        reg = ExperimentRegistry()
        for run_id in reg.list_by_algorithm(args.algorithm):
            print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
