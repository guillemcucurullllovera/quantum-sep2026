"""Evidence-level computation — the structural fix for the estate's one
verified failure mode (a hardware-validation claim citing a results file
that did not exist anywhere in the repo — see
`algorithms/qae_iqae/hardware/noise_aware.py`'s docstring history) and the
broader pattern QUANTUM-0 found repeatedly: unexecuted or unsupported
headline numbers presented as validated (`qrc_investor_demo.ipynb`, the
withdrawn IBM decks).

The evidence level of an `ExperimentResult` is never a free-typed field —
it is *computed* here from the result's own backend/shots/noise-model/
hardware-evidence fields, PLUS (for levels F and G) external state that a
bare `ExperimentResult` cannot manufacture on its own.

**Fail-closed against omission AND fabrication.** An earlier version of
this module was fail-closed only against omission: a missing `job_id`/
`backend_name`/`financial_backtest_artifact` raised, but a *fabricated*
(non-blank, well-formed, entirely made up) value sailed through to level F
or G with no cross-check against anything. An independent adversarial
review demonstrated this concretely. The fix, detailed in
`core.verification` and below:

- **Level F** requires `hardware_evidence.attestation_id` to resolve, via
  a caller-supplied `evidence_store`, to a `VerifiedAttestation` that only
  `core.verification.attest()` — by actually running the claim through a
  `HardwareEvidenceVerifier` — could have produced. No `evidence_store`
  supplied, or the id doesn't resolve, or the resolved record doesn't match
  the claimed fields ⇒ `EvidenceIntegrityError`, never a silent downgrade.
- **Level G** requires `financial_backtest_artifact` to resolve to a real
  artifact via a caller-supplied `result_store` (not merely a non-empty
  string), AND non-empty `classical_baseline_metrics`, AND that the
  claim's underlying execution (its A-F level) is itself substantiated —
  a financial-edge claim layered on top of an unverifiable QPU claim does
  not get to skip that verification.

Evidence ladder (A-F: how the result was computed; G: a distinct axis —
whether it is being presented as evidence of real trading/investment edge,
which is always the most demanding claim regardless of A-F level):

    A  analytical/theoretical   — proven by construction, no execution
    B  ideal simulator          — exact/statevector, no shot noise
    C  finite-shot ideal sim    — shot noise only, no device noise
    D  synthetic noise          — simulator + a non-device noise model
    E  device-derived noise     — fake-backend (e.g. FakeKingston) calibration snapshot
    F  real QPU                 — verified hardware execution, see above
    G  prospective financial evidence — claims real trading/investment edge;
       requires an executed, resolved backtest artifact, orthogonal to A-F

See `docs/quantum/QUANTUM1A_REVIEW_REMEDIATION.md` for what remains to be
implemented (a real `IBMRuntimeVerifier`, a persistent `EvidenceStore`)
before this gates an actual hardware run.
"""
from __future__ import annotations

from enum import Enum

from core.errors import EvidenceIntegrityError
from core.schemas import BackendKind, ExperimentResult


class EvidenceLevel(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


def compute_evidence_level(
    result: ExperimentResult,
    *,
    evidence_store: "object | None" = None,
    result_store: "object | None" = None,
) -> EvidenceLevel:
    """Derive the evidence level from `result`'s own recorded fields, plus
    (for F/G) the external stores that hold what was actually verified.

    Args:
        evidence_store: a `core.verification.EvidenceStore`-compatible
            object (`.get(attestation_id) -> VerifiedAttestation | None`).
            Required to reach level F. Deliberately has no default — a
            caller who omits it cannot accidentally fall back onto some
            globally-shared, possibly-stale store.
        result_store: a `core.result_store.ResultStore`-compatible object
            (`.artifact_exists(run_id, name) -> bool`). Required to reach
            level G for the same reason.

    Raises `EvidenceIntegrityError` (never silently downgrades) whenever a
    claim outruns what can actually be substantiated. See module docstring.
    """
    base_level = _compute_base_level(result, evidence_store=evidence_store)

    if result.financial_edge_claimed:
        _require_financial_evidence(result, result_store=result_store)
        return EvidenceLevel.G

    return base_level


def _compute_base_level(
    result: ExperimentResult, *, evidence_store: "object | None"
) -> EvidenceLevel:
    backend = result.config.backend

    if backend.kind == BackendKind.QPU:
        ev = result.hardware_evidence
        if ev is None or not ev.job_id or not ev.backend_name:
            raise EvidenceIntegrityError(
                f"run {result.metadata.run_id}: backend.kind=QPU but no auditable "
                "hardware_evidence (job_id, backend_name) is attached — cannot claim "
                "evidence level F without it."
            )
        if ev.backend_name != backend.name:
            raise EvidenceIntegrityError(
                f"run {result.metadata.run_id}: hardware_evidence.backend_name "
                f"({ev.backend_name!r}) does not match config.backend.name "
                f"({backend.name!r})."
            )
        if not ev.attestation_id:
            raise EvidenceIntegrityError(
                f"run {result.metadata.run_id}: backend.kind=QPU but "
                "hardware_evidence.attestation_id is unset. CLAIMED metadata "
                "(job_id/backend_name/execution_timestamp) is NEVER sufficient "
                "on its own for evidence level F — obtain a real attestation "
                "via core.verification.attest(claimed, verifier, store) first."
            )
        if evidence_store is None:
            raise EvidenceIntegrityError(
                f"run {result.metadata.run_id}: no evidence_store was supplied "
                "to compute_evidence_level, so hardware_evidence.attestation_id "
                "cannot be resolved. A serialized ExperimentResult is never "
                "sufficient on its own to establish evidence level F."
            )
        attestation = evidence_store.get(ev.attestation_id)
        if attestation is None:
            raise EvidenceIntegrityError(
                f"run {result.metadata.run_id}: hardware_evidence.attestation_id "
                f"{ev.attestation_id!r} does not resolve in the supplied "
                "evidence_store — this run was never verified, or the "
                "attestation id was typed by hand rather than produced by "
                "core.verification.attest()."
            )
        if attestation.backend_name != ev.backend_name or attestation.job_id != ev.job_id:
            raise EvidenceIntegrityError(
                f"run {result.metadata.run_id}: the resolved attestation's "
                f"backend_name/job_id ({attestation.backend_name!r}/"
                f"{attestation.job_id!r}) does not match the claimed fields "
                f"({ev.backend_name!r}/{ev.job_id!r}) — possible tampering "
                "with hardware_evidence after the attestation was recorded."
            )
        return EvidenceLevel.F

    if backend.kind == BackendKind.FAKE:
        return EvidenceLevel.E

    # BackendKind.SIMULATOR
    if result.config.noise_model is not None:
        return EvidenceLevel.D
    if result.config.shots:
        return EvidenceLevel.C
    return EvidenceLevel.B


def _require_financial_evidence(
    result: ExperimentResult, *, result_store: "object | None"
) -> None:
    if not result.financial_backtest_artifact:
        raise EvidenceIntegrityError(
            f"run {result.metadata.run_id}: financial_edge_claimed=True but "
            "financial_backtest_artifact is unset — a financial-edge claim "
            "requires an executed backtest artifact, not an asserted number."
        )
    if not result.classical_baseline_metrics:
        raise EvidenceIntegrityError(
            f"run {result.metadata.run_id}: financial_edge_claimed=True but "
            "classical_baseline_metrics is empty — a financial-edge claim "
            "must be run against the same classical comparator it claims to "
            "beat, not asserted in isolation."
        )
    if result_store is None:
        raise EvidenceIntegrityError(
            f"run {result.metadata.run_id}: no result_store was supplied to "
            "compute_evidence_level, so financial_backtest_artifact cannot "
            "be resolved. A non-empty string alone is never sufficient for "
            "evidence level G."
        )
    if "/" not in result.financial_backtest_artifact:
        raise EvidenceIntegrityError(
            f"run {result.metadata.run_id}: financial_backtest_artifact "
            f"{result.financial_backtest_artifact!r} is not a "
            "'{run_id}/{artifact_name}' reference resolvable through "
            "core.result_store — arbitrary non-empty text is not evidence."
        )
    run_id, name = result.financial_backtest_artifact.split("/", 1)
    if not result_store.artifact_exists(run_id, name):
        raise EvidenceIntegrityError(
            f"run {result.metadata.run_id}: financial_backtest_artifact "
            f"{result.financial_backtest_artifact!r} does not resolve to a "
            "real artifact in the supplied result_store — a plausible-looking "
            "path alone is not evidence."
        )


__all__ = ["EvidenceLevel", "compute_evidence_level"]
