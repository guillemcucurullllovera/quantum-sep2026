"""M3 (mthree) readout-error mitigation on the QAE counts path.

M3 (matrix-free measurement mitigation, Nation et al.) corrects readout/assignment error
on the measured bitstrings. We calibrate on the run backend's qubits, correct the raw
counts to a quasi-distribution, take the nearest valid probability distribution, and
marginalise the ancilla. mthree is Apache-2.0 (imported, not vendored).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class M3Result:
    p1_raw: float
    p1_mitigated: float
    successes_mitigated: int
    shots: int
    avg_readout_fidelity: float
    readout_fidelities: tuple
    mapping: Dict[int, int]


def _p1_from_prob(prob: Dict, ancilla_clbit: int, nbits: int) -> float:
    p1 = 0.0
    for key, val in prob.items():
        if isinstance(key, int):
            bit = (key >> ancilla_clbit) & 1
        else:
            s = str(key).replace(" ", "")
            idx = len(s) - 1 - ancilla_clbit
            bit = 1 if (0 <= idx < len(s) and s[idx] == "1") else 0
        if bit == 1:
            p1 += float(val)
    return p1


def _p1_from_counts(counts: Dict[str, int], ancilla_clbit: int) -> float:
    ones = total = 0
    for b, c in counts.items():
        s = b.replace(" ", "")
        total += c
        idx = len(s) - 1 - ancilla_clbit
        if 0 <= idx < len(s) and s[idx] == "1":
            ones += c
    return ones / total if total else 0.0


def apply_m3(
    backend,
    counts: Dict[str, int],
    mapping: Dict[int, int],
    shots: int,
    *,
    ancilla_clbit: int = 2,
    calibrate: bool = True,
    mitigator=None,
) -> M3Result:
    """Apply M3 readout mitigation to `counts` measured on `mapping` (clbit->physical qubit)."""
    import mthree

    if mitigator is None:
        mitigator = mthree.M3Mitigation(backend)
        if calibrate:
            mitigator.cals_from_system(list(mapping.values()))

    quasi = mitigator.apply_correction(counts, list(mapping.values()))
    prob = quasi.nearest_probability_distribution()

    nbits = max(mapping.keys()) + 1
    p1_mit = _p1_from_prob(prob, ancilla_clbit, nbits)
    p1_raw = _p1_from_counts(counts, ancilla_clbit)

    try:
        fids = [float(f) for f in mitigator.readout_fidelity(list(mapping.values()))]
        avg_fid = sum(fids) / len(fids) if fids else float("nan")
    except Exception:
        fids, avg_fid = [], float("nan")

    return M3Result(
        p1_raw=p1_raw, p1_mitigated=p1_mit,
        successes_mitigated=int(round(p1_mit * shots)), shots=int(shots),
        avg_readout_fidelity=avg_fid, readout_fidelities=tuple(fids),
        mapping={int(k): int(v) for k, v in mapping.items()},
    )


def build_mitigator(backend, mapping: Dict[int, int]):
    import mthree
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(list(mapping.values()))
    return mit
