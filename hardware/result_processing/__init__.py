"""Result normalization: `hardware.ibm_runtime.primitives.run_sampler` returns
counts in Qiskit's native convention (keys are little-endian bitstrings —
i.e. the leftmost character is the *highest-indexed* qubit, matching the
convention already documented and relied on in
`algorithms.pqk.noise_study` — see that module for where this convention
was originally pinned down after caught a qubit-ordering bug from
getting it wrong). This module is the one place that turns raw counts into
normalized, `core.schemas.ExperimentResult`-ready metrics, so every caller
gets the same probabilities/marginal/bit-order handling instead of each
reimplementing it (and risking the same class of bug again).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalizedResult:
    probabilities: dict[str, float]
    """counts, converted to probabilities; keys remain Qiskit's native
    little-endian bitstrings — NOT reversed here, callers that need
    big-endian / natural qubit-index order should use `to_qubit_marginals`."""
    shots: int
    backend_name: str
    most_likely_bitstring: str
    most_likely_probability: float


def normalize_counts(counts: dict[str, int], backend_name: str) -> NormalizedResult:
    shots = sum(counts.values())
    if shots == 0:
        raise ValueError("normalize_counts got a counts dict with zero total shots")
    probs = {k: v / shots for k, v in counts.items()}
    best_bits, best_p = max(probs.items(), key=lambda kv: kv[1])
    return NormalizedResult(
        probabilities=probs, shots=shots, backend_name=backend_name,
        most_likely_bitstring=best_bits, most_likely_probability=best_p,
    )


def to_qubit_marginals(counts: dict[str, int]) -> dict[int, float]:
    """P(qubit_i == '1') for every logical qubit index i, correctly
    accounting for Qiskit's little-endian bitstring convention: bitstring
    character at position `len(bitstring) - 1 - i` is qubit i's outcome.
    """
    shots = sum(counts.values())
    if shots == 0:
        return {}
    n = len(next(iter(counts)))
    marginals = {i: 0.0 for i in range(n)}
    for bitstring, c in counts.items():
        for i in range(n):
            if bitstring[len(bitstring) - 1 - i] == "1":
                marginals[i] += c
    return {i: v / shots for i, v in marginals.items()}


def to_experiment_metrics(result: NormalizedResult) -> dict[str, float]:
    """Flatten a NormalizedResult into the flat float-valued dict shape
    `core.schemas.ExperimentResult.metrics` expects."""
    return {
        "shots": float(result.shots),
        "most_likely_probability": result.most_likely_probability,
        "num_distinct_outcomes": float(len(result.probabilities)),
    }


__all__ = ["NormalizedResult", "normalize_counts", "to_qubit_marginals", "to_experiment_metrics"]


def __getattr__(name: str):
    # Lazy re-export: hardware.result_processing.braket_normalize imports
    # `NormalizedResult`/`normalize_counts` FROM this module, so importing
    # it eagerly at module load here would be circular. PEP 562 module
    # __getattr__ defers the import until one of these names is actually
    # accessed via `hardware.result_processing.<name>`.
    if name in _BRAKET_EXPORTS:
        from hardware.result_processing import braket_normalize

        return getattr(braket_normalize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_BRAKET_EXPORTS = (
    "braket_counts_to_qiskit_convention",
    "normalize_braket_local_result",
    "normalize_braket_task_result",
)
__all__ = list(__all__) + list(_BRAKET_EXPORTS)
