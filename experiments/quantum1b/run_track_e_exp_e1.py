"""Track E runner: EXP-E1 -- the standing classical-competitor gate,
applied to the platform's two existing QAE finance integrands (bull-call
spread, crash barrier). See
experiments/quantum1b/classical_competitor_gate.py for the full
methodology and docs/quantum/QUANTUM1B_TRACK_E_AE_VS_MC.md for the
write-up.

Entirely classical (no quantum circuits) -- fast, safe to run without any
resource discipline beyond the convention already applied throughout this
package.

Run with: `python3 -m experiments.quantum1b.run_track_e_exp_e1` from the
repo root.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from experiments.quantum1b.classical_competitor_gate import evaluate_classical_competitor_gate

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_e_exp_e1.json"


def _result_to_dict(r) -> dict:
    d = asdict(r)
    d["overall_verdict"] = r.overall_verdict.value
    d["checks"] = [
        {**asdict(c), "verdict": c.verdict.value} for c in r.checks
    ]
    return d


def main() -> dict:
    print("Running EXP-E1 (standing classical-competitor gate)...")

    # The gate's thresholds are fixed; qmc_kwargs/is_kwargs select which
    # book the audit is applied to. The defaults (bull-call-spread for
    # qmc, crash-barrier-shaped importance-sampling scenario for is) match
    # erosion_audit.py's own existing default book -- this IS the
    # reproduction-of-the-known-finding run, promoted to a gate verdict.
    default_book = evaluate_classical_competitor_gate(integrand_label="default_book (BS call spread + crash barrier)", seed=0)
    print(default_book.summary())

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "EXP-E1": {
            "default_book": _result_to_dict(default_book),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote results manifest to {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
