"""python -m tools.experiment_launch --config path/to/config.json

One-command experiment launcher: loads an ExperimentConfig from a JSON
file, dispatches through core.registry.ALGORITHM_REGISTRY via the generic
runner, and records the result to the experiment registry -- the CLI
front door to `experiments.runners.generic_runner`.
"""
from __future__ import annotations

import argparse
import sys

from experiments.configs.loader import load_experiment_config
from experiments.runners.generic_runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.experiment_launch")
    parser.add_argument("--config", required=True, help="path to an ExperimentConfig JSON file")
    parser.add_argument("--seed", type=int, default=None, help="override the config's seed")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    print(f"Launching experiment '{config.name}' (algorithm={config.algorithm})...")
    try:
        result = run_experiment(config)
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    # run_experiment already records to the registry; just report the id.
    print(f"Done. run_id={result.metadata.run_id}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
