"""Readout-error mitigation via mthree (matrix-free measurement mitigation,
Nation et al. — Apache-2.0, imported as a normal dependency, not vendored
or adapted; no attribution obligation beyond ordinary library use).

New module (not migrated) — Generation 2's QAE work used mthree directly
inline; this is the generic, algorithm-agnostic wrapper other algorithms
should call instead of each reimplementing calibration handling.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class M3Result:
    raw_counts: dict[str, int]
    mitigated_quasi_probabilities: dict[str, float]
    mapping: list[int]


class M3Mitigator:
    """Thin wrapper around mthree's M3 calibration + correction, producing
    typed results instead of raw mthree objects at every call site."""

    def __init__(self, backend, qubits: list[int]) -> None:
        import mthree

        self.backend = backend
        self.qubits = qubits
        self._mit = mthree.M3Mitigation(backend)
        self._calibrated = False

    def calibrate(self) -> None:
        self._mit.cals_from_system(self.qubits)
        self._calibrated = True

    def apply(self, counts: dict[str, int]) -> M3Result:
        if not self._calibrated:
            self.calibrate()
        quasi = self._mit.apply_correction(counts, self.qubits)
        # mthree's QuasiDistribution supports dict-style access; normalize
        # to a plain dict of bitstring -> probability for downstream use.
        mitigated = {k: float(v) for k, v in quasi.items()}
        return M3Result(raw_counts=counts, mitigated_quasi_probabilities=mitigated, mapping=self.qubits)


__all__ = ["M3Mitigator", "M3Result"]
