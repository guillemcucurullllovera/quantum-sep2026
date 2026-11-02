"""python -m tools.validation --algorithm qaoa_qubo

Runs the test subset relevant to a given algorithm and reports pass/fail
in a structured way -- useful before trusting a newly-registered or
newly-modified algorithm's output.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ALGO_TEST_PATHS = {
    "qaoa_qubo": ["tests/algorithms/test_qaoa_qubo.py", "tests/parity/test_qaoa_parity.py"],
    "qae_iqae": ["tests/algorithms/test_qae_iqae.py"],
    "qrc": ["tests/algorithms/test_qrc.py", "tests/parity/test_qrc_parity.py"],
    "pqk_v2": ["tests/algorithms/test_pqk.py", "tests/parity/test_kernel_parity.py"],
    "vqc_qml": ["tests/algorithms/test_vqc_qml.py"],
}


def validate(algorithm: str) -> tuple[bool, str]:
    paths = _ALGO_TEST_PATHS.get(algorithm)
    if not paths:
        return False, f"No known test mapping for algorithm={algorithm!r}. Known: {sorted(_ALGO_TEST_PATHS)}"

    existing = [p for p in paths if Path(p).exists()]
    if not existing:
        return False, f"None of the mapped test files exist on disk: {paths}"

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *existing, "-q"],
        capture_output=True,
        text=True,
    )
    passed = proc.returncode == 0
    return passed, proc.stdout[-2000:] + proc.stderr[-500:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.validation")
    parser.add_argument("--algorithm", required=True)
    args = parser.parse_args(argv)

    passed, output = validate(args.algorithm)
    print(output)
    print(f"\nVALIDATION {'PASSED' if passed else 'FAILED'} for {args.algorithm!r}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
