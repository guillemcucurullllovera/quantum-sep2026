"""Regenerates `frozen_controls_snapshot.json` from the CURRENT code.

**Added 2026-10-01, remediating independent-review finding P1-5** ("'Frozen'
classical controls are not actually frozen; the drift-detection mechanism
cannot catch the failure mode it is named for"). Running this script and
committing its output is now the ONLY way `frozen_controls_snapshot.json`
changes -- it is never written automatically by importing
`frozen_controls.py` or by running the test suite.

Usage: run this deliberately, review the resulting diff to
`frozen_controls_snapshot.json` like any other change, and commit it --
exactly the same discipline as any other change to a frozen/pinned
scientific artifact in this repository. Running it without reviewing the
diff defeats the entire point of a committed snapshot.
"""
from __future__ import annotations

from experiments.quantum1b.hardware_readiness.frozen_controls import (
    FROZEN_CONTROLS,
    _SNAPSHOT_PATH,
    write_snapshot,
)


def main() -> None:
    write_snapshot(FROZEN_CONTROLS, _SNAPSHOT_PATH)
    print(f"Wrote {len(FROZEN_CONTROLS)} frozen control(s) to {_SNAPSHOT_PATH}")
    for c in FROZEN_CONTROLS:
        print(
            f"  {c.family}/{c.n_candidates}/seed={c.seed}: "
            f"fingerprint={c.expected_fingerprint[:12]}... "
            f"n_qubits={c.expected_n_qubits} exact_optimum={c.expected_exact_optimum}"
        )


if __name__ == "__main__":
    main()
