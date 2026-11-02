"""Backend-level scoring: given a circuit's qubit requirement, rank the
catalog's fake backends by expected execution quality (best achievable
qubit-selection score, normalized by qubit count so a bigger noisier chip
doesn't automatically win over a smaller cleaner one).
"""
from __future__ import annotations

from dataclasses import dataclass

from hardware.backend_selection.capabilities import (
    BackendCapabilities,
    capabilities_from_braket_local_simulator,
    capabilities_from_ibm_backend,
)
from hardware.backends.catalog import BackendCatalog, ResolvedBackend
from hardware.qubit_selection.selector import QubitSelector


@dataclass
class BackendScore:
    backend_name: str
    num_qubits: int
    best_layout: list[int]
    layout_report: dict
    composite_score: float


def score_backend_for_circuit(resolved: ResolvedBackend, n_qubits_needed: int) -> BackendScore:
    if resolved.descriptor.kind.value == "simulator":
        # noiseless simulator: no calibration data to score against, treat as
        # the ideal reference point rather than fabricating a numeric score.
        return BackendScore(
            backend_name=resolved.descriptor.name,
            num_qubits=resolved.descriptor.num_qubits,
            best_layout=list(range(n_qubits_needed)),
            layout_report={"note": "noiseless simulator — no calibration to score"},
            composite_score=float("inf"),
        )
    selector = QubitSelector(resolved.backend)
    layout = selector.select_layout(n_qubits_needed)
    report = selector.layout_report(layout)
    composite = report["t1_mean_us"] / (1.0 + 100 * report["readout_error_mean"] + 100 * report["mean_2q_error"])
    return BackendScore(
        backend_name=resolved.descriptor.name,
        num_qubits=resolved.descriptor.num_qubits,
        best_layout=layout,
        layout_report=report,
        composite_score=composite,
    )


def rank_backends_for_circuit(n_qubits_needed: int, candidates: list[str] | None = None) -> list[BackendScore]:
    """Score+rank every hardware-kind backend in the catalog (skips the
    noiseless simulator by default — pass candidates explicitly to include
    it) that has enough qubits for the given circuit.
    """
    names = candidates or [n for n in BackendCatalog.list_backends() if n != "noiseless"]
    scores = []
    for name in names:
        resolved = BackendCatalog.get(name)
        if resolved.descriptor.num_qubits < n_qubits_needed:
            continue
        scores.append(score_backend_for_circuit(resolved, n_qubits_needed))
    return sorted(scores, key=lambda s: -s.composite_score)


@dataclass
class CrossCloudBackendScore:
    """Same ranking concept as `BackendScore`, but reported through the
    cloud-agnostic `BackendCapabilities` shape so IBM fake backends and
    Braket's local simulator can sit in one ranked list — see
    `rank_all_clouds_for_circuit`."""

    backend_name: str
    cloud: str
    num_qubits: int
    composite_score: float
    capabilities: BackendCapabilities
    note: str | None = None


def score_braket_local_simulator(n_qubits_needed: int) -> CrossCloudBackendScore:
    """Score Braket's local simulator for a circuit needing `n_qubits_needed`
    qubits. Like the noiseless Aer simulator in `score_backend_for_circuit`,
    there is no per-qubit calibration data for a noiseless local simulator
    to score against, so it is treated as an ideal reference point
    (`composite_score = inf`) rather than assigned a fabricated number.
    """
    caps = capabilities_from_braket_local_simulator()
    return CrossCloudBackendScore(
        backend_name=caps.name,
        cloud=caps.cloud,
        num_qubits=caps.num_qubits,
        composite_score=float("inf"),
        capabilities=caps,
        note="noiseless local simulator — no calibration to score against; treated as ideal reference point",
    )


def rank_all_clouds_for_circuit(
    n_qubits_needed: int, *, include_ibm_noiseless: bool = False
) -> list[CrossCloudBackendScore]:
    """Cross-cloud ranking: every IBM fake backend in the catalog
    with enough qubits, plus Braket's local simulator, scored on one
    comparable scale and returned sorted best-first.

    IBM fake backends keep the exact composite-score formula
    `score_backend_for_circuit` already uses (real T1/readout/2Q-error
    calibration data via `QubitSelector`) — this function does not
    reimplement that scoring, it wraps it in the cloud-agnostic
    `CrossCloudBackendScore` shape and merges in the Braket entry so a
    caller can rank across clouds without branching on which SDK produced
    each candidate.
    """
    scores: list[CrossCloudBackendScore] = []
    ibm_names = [n for n in BackendCatalog.list_backends() if n != "noiseless"]
    if include_ibm_noiseless:
        ibm_names = ["noiseless"] + ibm_names
    for name in ibm_names:
        resolved = BackendCatalog.get(name)
        if resolved.descriptor.num_qubits < n_qubits_needed:
            continue
        scored = score_backend_for_circuit(resolved, n_qubits_needed)
        caps = capabilities_from_ibm_backend(resolved)
        scores.append(
            CrossCloudBackendScore(
                backend_name=scored.backend_name,
                cloud=caps.cloud,
                num_qubits=scored.num_qubits,
                composite_score=scored.composite_score,
                capabilities=caps,
                note=scored.layout_report.get("note"),
            )
        )
    braket_score = score_braket_local_simulator(n_qubits_needed)
    if braket_score.num_qubits >= n_qubits_needed:
        scores.append(braket_score)
    return sorted(scores, key=lambda s: -s.composite_score)


__all__ = [
    "BackendScore",
    "score_backend_for_circuit",
    "rank_backends_for_circuit",
    "CrossCloudBackendScore",
    "score_braket_local_simulator",
    "rank_all_clouds_for_circuit",
]
