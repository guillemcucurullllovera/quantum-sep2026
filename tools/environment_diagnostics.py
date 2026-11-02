"""python -m tools.environment_diagnostics — health-check summary of the
platform's runtime environment: package versions, backend availability,
credential *presence* (never values).
"""
from __future__ import annotations

import argparse
import json
import sys

from core.config import get_settings
from core.provenance import environment_fingerprint


def diagnose() -> dict:
    settings = get_settings(refresh=True)
    fp = environment_fingerprint()

    fake_backends_ok = True
    fake_backend_error = None
    try:
        from hardware.fake_backends import list_fake_backends

        n_fake = len(list_fake_backends())
    except Exception as e:  # pragma: no cover - diagnostic path
        fake_backends_ok = False
        fake_backend_error = str(e)
        n_fake = 0

    return {
        "python": fp["python"],
        "platform": fp["platform"],
        "packages": fp["packages"],
        "ibm_quantum_token_present": settings.ibm.is_present,
        "ibm_quantum_channel": settings.ibm.channel,
        "data_dir": str(settings.data_dir),
        "results_dir": str(settings.results_dir),
        "fake_backends_available": n_fake,
        "fake_backends_ok": fake_backends_ok,
        "fake_backend_error": fake_backend_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.environment_diagnostics")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = diagnose()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=== Environment Diagnostics ===")
        print(f"Python: {report['python']}  ({report['platform']})")
        for pkg, ver in report["packages"].items():
            print(f"  {pkg}: {ver}")
        print(f"IBM Quantum token present: {report['ibm_quantum_token_present']}")
        print(f"Fake backends available: {report['fake_backends_available']}")
        if not report["fake_backends_ok"]:
            print(f"  WARNING: {report['fake_backend_error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
