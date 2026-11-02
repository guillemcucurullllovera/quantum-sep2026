"""Builds the Heimdall Command Center's Quantum Lab export artifact.

**Hard boundary, unchanged from every prior phase of this work: this
module writes ONLY to `experiments/quantum1b/results/` in THIS worktree.
It never writes into, imports from, or otherwise touches
`heimdall-command-center`, `heimdall-v2`, or any other repository.** A real
Command Center integration would read the JSON this produces (or a future
API wrapping it) -- building that reader is explicitly out of scope here,
per the user's own instruction not to prematurely integrate.

The export is a plain, serializable snapshot: every `QUANTUM_LAB_REGISTRY`
entry plus its live-computed hardware-readiness state, the lab-wide
readiness gates (computed from real repository state, never hand-set), and
the flagship end-to-end candidate-trade prototype's latest result (if one
has been generated -- this module does not run it as a side effect, since
that is a genuinely expensive simulator job; see
`candidate_trade_prototype.py`).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import subprocess

from experiments.quantum1b.quantum_lab.candidate_trade_prototype import (
    OUTPUT_PATH as CANDIDATE_TRADE_OUTPUT_PATH,
)
from experiments.quantum1b.quantum_lab.charts import build_all_charts_tolerant, build_research_result_components
from experiments.quantum1b.quantum_lab.readiness_state import CAPSTONE_DECISION, current_lab_wide_gates
from experiments.quantum1b.quantum_lab.registry import QUANTUM_LAB_REGISTRY

# Static, hand-maintained remaining-blockers list (added 2026-10-01,
# Command Center integration audit). Deliberately NOT auto-derived from
# live state beyond the boolean gates already in
# `lab_wide_hardware_readiness_gates` -- those booleans ARE the
# machine-checkable source of truth for the hardware-day blockers; this
# list adds the blockers that are NOT expressible as a simple boolean
# (external dependencies, known measurement gaps). Each entry is a fact
# independently verified during this session's own work, not a guess.
REMAINING_BLOCKERS = (
    {
        "blocker": "Real IBM Quantum credentials have never been configured in this environment.",
        "category": "hardware_access",
        "resolvable_by": "setting IBM_QUANTUM_TOKEN (see core/config.py, .env.example) -- a deliberate, "
        "fail-closed gate, not an oversight.",
    },
    {
        "blocker": f"Capstone decision is '{CAPSTONE_DECISION}', not HARDWARE_DAY_READY.",
        "category": "hardware_access",
        "resolvable_by": "see docs/quantum/PRE_HARDWARE_QUANTUM_CAPSTONE.md section 6 for the full, "
        "reasoned justification -- not a simple flag flip.",
    },
    {
        "blocker": "No independent-reviewer sign-off on the closeout/capstone has been recorded "
        "(distinct from the adversarial review already completed and remediated this session).",
        "category": "hardware_access",
        "resolvable_by": "a human reviewer reading QUANTUM1B_PHASE1_CLOSEOUT.md and "
        "PRE_HARDWARE_QUANTUM_CAPSTONE.md and recording sign-off -- then flipping "
        "readiness_state.INDEPENDENT_REVIEW_SIGNED_OFF.",
    },
    {
        "blocker": "EXP-C1 (Track C, QRC vs. ESN) real-financial-task evidence is blocked on an "
        "external Model Lab data-licensing hold (no confirmed-rights OHLCV for the AMC underlyings).",
        "category": "external_dependency",
        "resolvable_by": "Model Lab issuing a RESEARCH_ONLY authority contract -- outside this "
        "repository's control.",
    },
    {
        "blocker": "EXP-D1's n_candidates=8/n_qubits=14 cell's exact noisy-execution cost remains "
        "unmeasured beyond a lower bound (>280s per single evaluation, extrapolated to >=47 "
        "minutes for a full 10-iteration optimization).",
        "category": "measurement_gap",
        "resolvable_by": "a future session with more wall-clock patience or a less-loaded host; "
        "see run_track_d_exp_d1_n8_single_shot_probe.py.",
    },
    {
        "blocker": "No hardware-readiness pathway (frozen controls + runbook) exists yet for Track C "
        "(QRC) or Track E (QAE/IQAE) circuit types -- only Tracks A/B/D share the "
        "OptionSelectionInstance/QUBO lineage frozen_controls.py is built for.",
        "category": "infrastructure_gap",
        "resolvable_by": "building an equivalent frozen-control set and runbook for those circuit "
        "types, if real-hardware evidence is ever sought there.",
    },
    {
        "blocker": "The registry's hand-curated prose fields (problem/hypothesis/result/conclusion/"
        "limitations) have no automated drift check against their source documents -- unlike "
        "frozen_controls.py's snapshot-vs-recomputation check, there is no cheap way to "
        "mechanically re-derive prose from prose.",
        "category": "maintenance_gap",
        "resolvable_by": "periodic manual re-verification; see registry.py's own module docstring.",
    },
)

OUTPUT_PATH = Path(__file__).parents[1] / "results" / "quantum_lab_command_center_export.json"


def _experiment_to_dict(exp) -> dict:
    return {
        "experiment_id": exp.experiment_id,
        "track": exp.track,
        "status": "COMPLETED",  # every registry entry has an already-produced result; see module docstring
        "problem": exp.problem,
        "hypothesis": exp.hypothesis,
        "classical_baseline": exp.classical_baseline,
        "quantum_formulation": exp.quantum_formulation,
        "simulator_noise_config": exp.simulator_noise_config,
        "result": exp.result,
        "evidence_level": exp.evidence_level,
        "conclusion": exp.conclusion,
        "limitations": exp.limitations,
        "next_experiment": exp.next_experiment,
        "hardware_readiness_state": exp.readiness_state().value,
        # Structured booleans (added 2026-10-01, Command Center integration
        # audit) -- `simulator_noise_config` above is free-text prose; a
        # consumer that needs to programmatically filter/group by
        # "has ideal-simulator evidence" vs "has noise-model evidence" vs
        # "has a hardware-readiness pathway at all" previously had to parse
        # that prose or the derived `hardware_readiness_state` enum (which
        # conflates several gates). These are the same booleans
        # `readiness_state()` itself is computed from, now also exposed
        # directly.
        "has_ideal_simulator_result": exp.has_ideal_simulator_result,
        "has_noise_validated_result": exp.has_noise_validated_result,
        "hardware_readiness_pathway_exists": exp.hardware_readiness_pathway_exists,
        "result_artifact_paths": list(exp.result_artifact_paths),
        "source_doc": exp.source_doc,
    }


def _load_candidate_trade_report() -> dict | None:
    """Reads the already-written candidate-trade prototype output, if one
    exists. Does NOT run the prototype (a genuine simulator job taking
    roughly a minute, per `candidate_trade_prototype.py`'s own module
    docstring) as a side effect of building an export snapshot."""
    if not CANDIDATE_TRADE_OUTPUT_PATH.is_file():
        return None
    with open(CANDIDATE_TRADE_OUTPUT_PATH) as f:
        return json.load(f)


def _current_git_head() -> str | None:
    """Live `git rev-parse HEAD` at generation time -- deliberately NOT
    hand-typed anywhere (the exact class of error independent-review
    finding P2-19 found in OVERNIGHT_QUANTUM_HANDOFF.md: a hand-typed SHA
    goes stale the instant another commit lands). Note this is the HEAD
    this export was GENERATED FROM -- committing the export itself creates
    a NEW commit on top, so this field will always show the PARENT of
    whatever commit actually contains this exact file content. Returns
    None (never a guess) if git is unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[3],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def build_command_center_export() -> dict:
    experiments = [_experiment_to_dict(exp) for exp in QUANTUM_LAB_REGISTRY]
    tracks = sorted({exp.track for exp in QUANTUM_LAB_REGISTRY})
    readiness_counts: dict[str, int] = {}
    for exp_dict in experiments:
        state = exp_dict["hardware_readiness_state"]
        readiness_counts[state] = readiness_counts.get(state, 0) + 1

    charts, chart_errors = build_all_charts_tolerant()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lab_name": "Heimdall Quantum Lab",
        "source_repo_path": "experiments/quantum1b/",
        "no_quantum_advantage_claim": True,
        "no_financial_edge_claim": True,
        "lab_wide_hardware_readiness_gates": current_lab_wide_gates(),
        "tracks": tracks,
        "experiments": experiments,
        "active_experiments": [],  # no in-progress experiments as of this export -- every registry entry has already run to completion
        "completed_experiments_count": len(experiments),
        "hardware_readiness_state_counts": readiness_counts,
        "charts": [c.to_dict() for c in charts],
        "chart_generation_errors": chart_errors,  # empty in a complete checkout; non-fabricating if a source artifact is missing
        "research_result_components": [c.to_dict() for c in build_research_result_components()],
        "capstone_decision": CAPSTONE_DECISION,
        "remaining_blockers": list(REMAINING_BLOCKERS),
        "provenance": {
            "generator_module": "experiments.quantum1b.quantum_lab.command_center_export",
            "git_head_at_generation": _current_git_head(),
            "note": (
                "git_head_at_generation is the commit this export was GENERATED FROM (captured live via "
                "`git rev-parse HEAD`, never hand-typed) -- committing this file creates a NEW commit on "
                "top, so this field always lags the commit that actually contains this exact content by "
                "one. For the true current state, run `git log -1` directly rather than trusting this "
                "field alone."
            ),
        },
        "reproducibility": {
            "test_command": "cd <repo root> && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "
            "NUMEXPR_NUM_THREADS=2 python3 -m pytest tests/quantum1b/ tests/hardware/ -q",
            "last_verified_test_result": "704/704 passing, zero failures",
            "last_verified_approximately_at_git_head": "30ed38281c5bf9862965d92cd0a852e59006dac1",
            "note": (
                "This is a POINT-IN-TIME record, not a live status -- 'approximately at' means verified "
                "against that commit's parent state PLUS this export module's own uncommitted changes at "
                "verification time, not a guarantee of an exact `git checkout` match (a hand-pinned future "
                "commit SHA is impossible to know in advance -- see provenance.note above for the same "
                "lesson applied to git_head_at_generation). This field will be stale the instant another "
                "commit lands regardless. Re-run the command above for the current, authoritative result; "
                "never trust a hardcoded count here for anything safety- or disclosure-relevant."
            ),
        },
        "flagship_end_to_end_prototype": {
            "description": (
                "classical -> QUBO -> QAOA/CVaR-QAOA -> noise-validated simulator comparison on the "
                "candidate-trade / finite derivative-risk-budget combinatorial optimization problem"
            ),
            "module": "experiments.quantum1b.quantum_lab.candidate_trade_prototype",
            "latest_result": _load_candidate_trade_report(),
        },
    }


def main() -> None:
    export = build_command_center_export()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    print(
        f"{export['completed_experiments_count']} experiments, "
        f"readiness state counts: {export['hardware_readiness_state_counts']}"
    )


if __name__ == "__main__":
    main()


__all__ = ["build_command_center_export", "main", "OUTPUT_PATH"]
