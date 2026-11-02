"""Backend/job/result lineage + circuit/transpilation metadata capture.

Reuses, rather than reimplements:
- `algorithms.qae_iqae.backends.transpile_for` / `TranspiledCircuit` for
  transpilation metadata (depth, gate counts, two-qubit gate count) --
  already built, already used for the QAE pipeline's FakeKingston work.
- `algorithms.qaoa_qubo.solvers._build_qiskit_qaoa_circuit` for the actual
  QAOA circuit construction -- already the exact same standard-mixer
  convention `experiments.quantum1b.cvar_qaoa` documents itself as
  matching.

This module's only net-new contribution is `RunLineageRecord`: a single,
serializable record tying together (instance fingerprint -> circuit
metadata -> backend descriptor -> claimed/verified hardware evidence),
so a later process (or a human, the morning after) can answer "what
actually produced this result" without cross-referencing five separate
files by hand.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from algorithms.qae_iqae.backends import TranspiledCircuit, transpile_for
from algorithms.qaoa_qubo.qubo_builder import qubo_to_ising
from algorithms.qaoa_qubo.solvers import _build_qiskit_qaoa_circuit
from core.schemas import BackendDescriptor, HardwareRunEvidence


def build_qaoa_circuit_for_lineage(Q: np.ndarray, p: int, gamma: np.ndarray, beta: np.ndarray):
    """Thin, documented wrapper around
    `algorithms.qaoa_qubo.solvers._build_qiskit_qaoa_circuit` -- imported
    directly (it is a private-by-convention but stable, already-tested
    function; wrapping it here gives hardware_readiness its own named,
    documented entry point rather than every caller reaching into
    `solvers.py`'s private namespace directly)."""
    J, h, _offset = qubo_to_ising(Q)
    return _build_qiskit_qaoa_circuit(J, h, p, gamma, beta)


def capture_transpilation_metadata(
    circuit, backend, *, optimization_level: int = 1, seed: int = 42
) -> TranspiledCircuit:
    """Thin wrapper around `algorithms.qae_iqae.backends.transpile_for` --
    same function, reused verbatim, given a name under this package so
    Track B/D callers don't need to know the metadata capture happens to
    live in the QAE package today."""
    return transpile_for(circuit, backend, optimization_level=optimization_level, seed=seed)


@dataclass(frozen=True)
class RunLineageRecord:
    """One run's full backend/job/result lineage, from the frozen problem
    instance through to (claimed, and once verified, attested) hardware
    evidence -- the single object a later process or reviewer should need
    to reconstruct "what actually happened" for one run."""

    instance_fingerprint: str
    qubo_n_qubits: int
    circuit_depth: int
    circuit_gate_counts: dict[str, int]
    circuit_two_qubit_gates: int
    backend: BackendDescriptor
    shots: int | None
    seed: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hardware_evidence: HardwareRunEvidence | None = None

    def record_hash(self) -> str:
        """A deterministic fingerprint of this lineage record's own
        content (excluding `created_at`, which is wall-clock and not part
        of the scientific content) -- lets two lineage records be compared
        for "same experimental setup" independent of when each ran.

        **Also excludes `hardware_evidence` (documented explicitly
        2026-10-01, remediating independent-review finding P3-8).** Two
        records with identical circuit/backend/shots/seed but completely
        different hardware jobs (or no hardware job at all vs. a real one)
        get an IDENTICAL `record_hash()` -- this is intentional (the hash
        identifies the EXPERIMENTAL SETUP, "what would this run have
        produced," not which specific hardware execution attempt it was),
        but it means this hash must never be treated as uniquely
        identifying a specific hardware run. Use `hardware_evidence`'s own
        `job_id`/`attestation_id` for that."""
        payload = {
            "instance_fingerprint": self.instance_fingerprint,
            "qubo_n_qubits": self.qubo_n_qubits,
            "circuit_depth": self.circuit_depth,
            "circuit_gate_counts": self.circuit_gate_counts,
            "circuit_two_qubit_gates": self.circuit_two_qubit_gates,
            "backend_name": self.backend.name,
            "backend_kind": self.backend.kind.value,
            "shots": self.shots,
            "seed": self.seed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["backend"] = json.loads(self.backend.model_dump_json())
        if self.hardware_evidence is not None:
            d["hardware_evidence"] = json.loads(self.hardware_evidence.model_dump_json())
        d["record_hash"] = self.record_hash()
        return d


def build_lineage_record(
    *,
    instance_fingerprint: str,
    transpiled: TranspiledCircuit,
    qubo_n_qubits: int,
    backend: BackendDescriptor,
    shots: int | None,
    seed: int,
    hardware_evidence: HardwareRunEvidence | None = None,
) -> RunLineageRecord:
    return RunLineageRecord(
        instance_fingerprint=instance_fingerprint,
        qubo_n_qubits=qubo_n_qubits,
        circuit_depth=transpiled.depth,
        circuit_gate_counts=transpiled.gate_counts,
        circuit_two_qubit_gates=transpiled.twoq_gates,
        backend=backend,
        shots=shots,
        seed=seed,
        hardware_evidence=hardware_evidence,
    )


__all__ = [
    "build_qaoa_circuit_for_lineage",
    "capture_transpilation_metadata",
    "RunLineageRecord",
    "build_lineage_record",
]
