"""Machine-readable preregistration registry.

Each `experiments/quantum1b/registry/EXP-*.json` mirrors a section of
docs/quantum/QUANTUM1B_PREREGISTRATION.md -- the prose is authoritative,
these are a structured, loadable/validated index of the same commitments
(hypothesis, metrics, seeds, stopping rule, compute budget, pass/fail
interpretation, evidence level) so a result runner can assert it is
following the preregistered plan rather than drifting from it silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY_DIR = Path(__file__).parent / "registry"


@dataclass(frozen=True)
class PreregisteredExperiment:
    experiment_id: str
    track: str  # "A" | "B" | "C"
    hypothesis: str
    benchmark_problem: str
    classical_comparator: str
    quantum_method: str | None
    metrics: tuple[str, ...]
    seeds: tuple[int, ...]
    stopping_rule: str
    compute_budget: str
    pass_fail_interpretation: dict[str, str]
    evidence_level: str
    prohibited_conclusion: str

    @staticmethod
    def from_dict(d: dict) -> "PreregisteredExperiment":
        return PreregisteredExperiment(
            experiment_id=d["experiment_id"],
            track=d["track"],
            hypothesis=d["hypothesis"],
            benchmark_problem=d["benchmark_problem"],
            classical_comparator=d["classical_comparator"],
            quantum_method=d.get("quantum_method"),
            metrics=tuple(d["metrics"]),
            seeds=tuple(d["seeds"]),
            stopping_rule=d["stopping_rule"],
            compute_budget=d["compute_budget"],
            pass_fail_interpretation=dict(d["pass_fail_interpretation"]),
            evidence_level=d["evidence_level"],
            prohibited_conclusion=d["prohibited_conclusion"],
        )


def load_registry(directory: Path = REGISTRY_DIR) -> dict[str, PreregisteredExperiment]:
    out: dict[str, PreregisteredExperiment] = {}
    for path in sorted(directory.glob("EXP-*.json")):
        with open(path) as f:
            d = json.load(f)
        exp = PreregisteredExperiment.from_dict(d)
        if exp.experiment_id in out:
            raise ValueError(f"duplicate experiment_id {exp.experiment_id} in registry")
        out[exp.experiment_id] = exp
    return out
