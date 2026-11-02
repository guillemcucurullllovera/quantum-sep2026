"""Estimate the total resource cost of running many circuits (a
"campaign") — the planning question "how long will this whole sweep take
and how many shots will it consume", not "how deep is this one circuit"
(that's `hardware.circuit_analysis`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hardware.circuit_analysis import CircuitStats, analyze

# Rough wall-clock model: per-shot execution time scales with circuit depth
# (more layers -> more gate-time), plus a fixed per-job overhead (compile +
# queue + result-fetch latency). These constants are simulator/local-queue
# estimates, not IBM's published per-backend numbers (which require a live
# service query) -- documented as such, not presented as measured fact.
_NS_PER_DEPTH_LAYER = 100.0  # nanoseconds of gate time per circuit-depth layer
_JOB_OVERHEAD_SECONDS = 2.0  # compile + submit + fetch overhead per job


@dataclass
class CircuitJob:
    name: str
    circuit: object
    shots: int


@dataclass
class CampaignEstimate:
    n_jobs: int
    total_shots: int
    total_circuit_seconds: float
    total_overhead_seconds: float
    total_estimated_seconds: float
    per_job: list[dict] = field(default_factory=list)

    @property
    def total_estimated_minutes(self) -> float:
        return self.total_estimated_seconds / 60.0


def estimate_campaign(jobs: list[CircuitJob]) -> CampaignEstimate:
    per_job = []
    total_shots = 0
    total_circuit_seconds = 0.0
    for job in jobs:
        stats: CircuitStats = analyze(job.circuit)
        circuit_seconds = (stats.depth * _NS_PER_DEPTH_LAYER * job.shots) / 1e9
        per_job.append(
            {
                "name": job.name,
                "shots": job.shots,
                "depth": stats.depth,
                "two_qubit_gate_count": stats.two_qubit_gate_count,
                "estimated_seconds": circuit_seconds + _JOB_OVERHEAD_SECONDS,
            }
        )
        total_shots += job.shots
        total_circuit_seconds += circuit_seconds

    total_overhead = _JOB_OVERHEAD_SECONDS * len(jobs)
    return CampaignEstimate(
        n_jobs=len(jobs),
        total_shots=total_shots,
        total_circuit_seconds=total_circuit_seconds,
        total_overhead_seconds=total_overhead,
        total_estimated_seconds=total_circuit_seconds + total_overhead,
        per_job=per_job,
    )
