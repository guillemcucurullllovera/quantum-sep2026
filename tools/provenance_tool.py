"""python -m tools.provenance_tool --run-id <id>

Dumps a full reproducibility record for a given run, using
hardware.provenance.build_reproducibility_record.
"""
from __future__ import annotations

import argparse
import json
import sys

from core.experiment_registry import ExperimentRegistry
from hardware.provenance import build_reproducibility_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.provenance_tool")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    registry = ExperimentRegistry()
    try:
        result = registry.get(args.run_id)
    except Exception as e:
        print(f"Could not find run {args.run_id!r}: {e}", file=sys.stderr)
        return 1

    record = build_reproducibility_record(result)
    print(json.dumps(record.model_dump(mode="json"), indent=2, default=str))
    if record.missing_fields:
        print(f"\nWARNING: missing fields for full reproducibility: {record.missing_fields}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
