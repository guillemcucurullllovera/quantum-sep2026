"""Typed domain models shared across algorithms, applications, and hardware.

Uses pydantic so that (de)serialization to/from the result store and
experiment registry is validated, not ad hoc dict-passing.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class BackendKind(str, Enum):
    SIMULATOR = "simulator"          # Aer statevector/qasm simulator, no noise
    FAKE = "fake"                     # Qiskit fake-backend (device-calibrated noise)
    QPU = "qpu"                       # real IBM hardware — requires credentials


class MitigationKind(str, Enum):
    NONE = "none"
    READOUT_M3 = "readout_m3"
    ZNE = "zne"


class BackendDescriptor(BaseModel):
    name: str
    kind: BackendKind
    num_qubits: int
    provider: str = "ibm"
    coupling_map: list[tuple[int, int]] | None = None
    basis_gates: list[str] | None = None
    calibration_source: str | None = Field(
        default=None,
        description="Where calibration data came from, e.g. 'fake_backend:FakeSherbrooke' "
        "or 'live:2026-08-30T12:00Z'. Never blank for QPU-kind backends.",
    )


class NoiseModelDescriptor(BaseModel):
    backend_name: str
    readout_error_mean: float | None = None
    gate_error_mean: float | None = None
    source: str = "fake_backend"


class MitigationConfig(BaseModel):
    kind: MitigationKind = MitigationKind.NONE
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetDescriptor(BaseModel):
    name: str
    n_samples: int
    n_features: int
    domain: str = Field(description="e.g. 'realized_volatility', 'directional_classification'")
    source_path: str | None = None
    checksum: str | None = None


class RunMetadata(BaseModel):
    run_id: str = Field(default_factory=_new_id)
    created_at: float = Field(default_factory=time.time)
    algorithm: str
    generation: str | None = Field(
        default=None, description="'gen1', 'gen2', or None for net-new work"
    )
    adapted_from: str | None = Field(
        default=None, description="Upstream project/paper citation if this is an adapted algorithm"
    )
    seed: int | None = None
    git_context: str | None = None


class ExperimentConfig(BaseModel):
    name: str
    algorithm: str
    application: str | None = None
    backend: BackendDescriptor
    mitigation: MitigationConfig = Field(default_factory=MitigationConfig)
    noise_model: NoiseModelDescriptor | None = Field(
        default=None,
        description="Set only when a SIMULATOR-kind backend has a noise model "
        "applied. Distinguishes evidence level D (synthetic noise) from B/C "
        "(ideal simulator) — see core.evidence.compute_evidence_level.",
    )
    shots: int | None = 4096
    seed: int = 42
    params: dict[str, Any] = Field(default_factory=dict)


class HardwareRunEvidence(BaseModel):
    """CLAIMED metadata for a real QPU execution — self-typed by whoever
    constructs the result, and therefore NEVER sufficient on its own to
    establish evidence level F. See `core.verification` for the separate
    VERIFIED side (an attestation produced only by running these claimed
    fields through a `HardwareEvidenceVerifier` against an external system
    of record) and `core.evidence.compute_evidence_level` for how the two
    are bound together.

    An independent adversarial review of QUANTUM-1A demonstrated that an
    earlier version of this model (checked only for non-blank `job_id`/
    `backend_name`, with `execution_timestamp` entirely unvalidated) let a
    caller construct `HardwareRunEvidence(backend_name="ibm_brisbane",
    job_id="totally-made-up-job-id", execution_timestamp="")` and receive
    evidence level F with zero cross-checks. The validators below close the
    *well-formedness* half of that gap (a blank/unparseable/future
    timestamp is now rejected at construction time, before evidence
    computation ever runs) — they do NOT and cannot prove `job_id`/
    `backend_name` correspond to a real job. That proof requires
    `attestation_id` to resolve against a `core.verification.EvidenceStore`
    entry that only a verifier could have written — see
    `core.evidence.compute_evidence_level`.
    """

    backend_name: str
    job_id: str
    execution_timestamp: str
    session_id: str | None = None
    attestation_id: str | None = Field(
        default=None,
        description="Set only via core.verification.attest(claimed, verifier, "
        "store) against a real HardwareEvidenceVerifier — never typed by "
        "hand. Required for evidence level F. A hand-typed value here will "
        "simply fail to resolve against any EvidenceStore and will not "
        "grant level F; see core.evidence.compute_evidence_level.",
    )

    @field_validator("job_id", "backend_name")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("execution_timestamp")
    @classmethod
    def _valid_non_future_timestamp(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("execution_timestamp must not be empty")
        try:
            dt = datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"execution_timestamp must be valid ISO-8601: {exc}") from exc
        if dt.tzinfo is None:
            raise ValueError("execution_timestamp must be timezone-aware")
        tolerance = timedelta(minutes=5)
        if dt > datetime.now(timezone.utc) + tolerance:
            raise ValueError(
                f"execution_timestamp {v!r} is in the future beyond the "
                f"documented {tolerance} clock-skew tolerance"
            )
        return v


class ExperimentResult(BaseModel):
    metadata: RunMetadata
    config: ExperimentConfig
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(
        default_factory=dict, description="artifact name -> result_store path"
    )
    classical_baseline_metrics: dict[str, float] | None = Field(
        default=None,
        description="Populated whenever a classical comparator was run alongside "
        "the quantum method, per the platform's benchmarking policy.",
    )
    hardware_evidence: HardwareRunEvidence | None = Field(
        default=None,
        description="Required whenever config.backend.kind == BackendKind.QPU. "
        "Never hand-asserted elsewhere — evidence level F is computed from "
        "this field's presence, not from a docstring or notes claim.",
    )
    financial_edge_claimed: bool = Field(
        default=False,
        description="Set only when this result is being presented as evidence "
        "that a method improves real trading/investment outcomes (evidence "
        "level G) — a strictly higher evidentiary bar than algorithmic "
        "correctness or simulator/hardware fidelity. Level G additionally "
        "requires non-empty classical_baseline_metrics AND "
        "financial_backtest_artifact to resolve to a real, existing artifact "
        "through core.result_store (a non-empty string alone is not "
        "evidence — see core.evidence.compute_evidence_level), or evidence "
        "computation fails closed. This is the direct fix for the withdrawn "
        "qrc_investor_demo.ipynb / IBM-deck pattern: unexecuted financial "
        "projections presented as validated.",
    )
    financial_backtest_artifact: str | None = Field(
        default=None,
        description="'{run_id}/{artifact_name}' addressing an artifact actually "
        "saved via core.result_store.ResultStore — resolved (not merely "
        "checked for non-emptiness) by core.evidence.compute_evidence_level "
        "before granting level G. Must be run against the same classical "
        "comparator recorded in classical_baseline_metrics.",
    )
    notes: str | None = None


class BenchmarkResult(BaseModel):
    benchmark_id: str = Field(default_factory=_new_id)
    created_at: float = Field(default_factory=time.time)
    method_a: str
    method_b: str
    metric_name: str
    value_a: float
    value_b: float
    wall_time_a_s: float | None = None
    wall_time_b_s: float | None = None
    circuit_depth: int | None = None
    two_qubit_gate_count: int | None = None
    shots: int | None = None
    seed: int | None = None
    context: dict[str, Any] = Field(default_factory=dict)
