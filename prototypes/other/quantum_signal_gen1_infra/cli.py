"""
CLI entry point for quantum signal vector builder.

Usage:
    python -m quantum_signal.cli --input-dir tests/fixtures/ --asset EURUSD --format json
    python -m quantum_signal.cli --input-dir tests/fixtures/ --asset EURUSD --format csv
    python -m quantum_signal.cli --input-dir tests/fixtures/ --asset EURUSD --format array
    python -m quantum_signal.cli --input-dir tests/fixtures/ --asset EURUSD --validate
"""

import argparse
import json
import sys
from pathlib import Path

from prototypes.other.quantum_signal_gen1_infra.builder import build_quantum_vector
from prototypes.other.quantum_signal_gen1_infra.validator import VectorValidator


def load_fixture(path: Path) -> dict:
    """Load a JSON fixture file."""
    with open(path) as f:
        return json.load(f)


def find_fixture(directory: Path, agent_type: str) -> dict:
    """Find and load a fixture file matching the agent type."""
    for pattern in [f"sample_{agent_type}.json", f"*{agent_type}*.json", f"{agent_type}.json"]:
        matches = list(directory.glob(pattern))
        if matches:
            return load_fixture(matches[0])
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Build a quantum input vector from raw agent signals."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing agent signal JSON files.",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="UNKNOWN",
        help="Asset identifier (e.g., EURUSD).",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="ISO 8601 timestamp. Defaults to now.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "array"],
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation and print report.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Load agent signals
    technical = find_fixture(input_dir, "technical")
    macro = find_fixture(input_dir, "macro")
    news = find_fixture(input_dir, "news")
    fundamental = find_fixture(input_dir, "fundamental")
    risk_sentiment = find_fixture(input_dir, "risk_sentiment")
    commodity = find_fixture(input_dir, "commodity")

    # Use timestamp from technical signal if not provided
    timestamp = args.timestamp or technical.get("timestamp")

    # Build vector
    vector = build_quantum_vector(
        technical=technical,
        macro=macro,
        news=news,
        fundamental=fundamental,
        risk_sentiment=risk_sentiment,
        commodity=commodity,
        asset=args.asset,
        timestamp=timestamp,
    )

    # Validate if requested
    if args.validate:
        validator = VectorValidator()
        report = validator.validate(vector)
        print(f"Valid: {report.is_valid}")
        if report.warnings:
            print(f"Warnings: {report.warnings}")
        if report.errors:
            print(f"Errors: {report.errors}")
        print()

    # Format output
    if args.format == "json":
        output = vector.to_json()
    elif args.format == "csv":
        fields = vector.to_dict()
        header = ",".join(fields.keys())
        values = ",".join(str(v) for v in fields.values())
        output = f"{header}\n{values}"
    elif args.format == "array":
        arr = vector.to_array()
        output = json.dumps(arr)
    else:
        output = vector.to_json()

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
