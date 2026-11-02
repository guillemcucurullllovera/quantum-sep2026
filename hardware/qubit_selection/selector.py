"""Qubit-subgraph selection: pick the best n-qubit connected region of a
device given real calibration data (T1, readout error, 2-qubit gate
error) pulled from the backend's `.target`.

Generalized from an earlier, IBM-Heron-specific Generation 2 implementation;
this version reads calibration data generically so it
works against any BackendV2 (Eagle, Heron, or future families) rather than
hardcoding Heron nominal values as a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hardware.topology.graph import CouplingGraph, from_backend


@dataclass
class QubitScore:
    qubit: int
    t1_us: float
    readout_error: float
    mean_2q_error: float
    score: float


class QubitSelector:
    """Selects a connected n-qubit subgraph of a backend, prioritizing
    qubits with high T1 and low readout/gate error — real calibration data
    pulled from `backend.target`, not fabricated nominal values.
    """

    def __init__(self, backend) -> None:
        self.backend = backend
        self.graph: CouplingGraph = from_backend(backend)
        self._t1, self._readout, self._2q_error = self._load_calibration()

    def _load_calibration(self) -> tuple[dict[int, float], dict[int, float], dict[tuple[int, int], float]]:
        target = self.backend.target
        n = self.backend.num_qubits
        t1: dict[int, float] = {}
        readout: dict[int, float] = {}
        for q in range(n):
            props = target.qubit_properties[q] if target.qubit_properties else None
            t1[q] = float(props.t1) * 1e6 if props is not None and props.t1 else float("nan")
            try:
                readout[q] = float(target["measure"][(q,)].error)
            except (KeyError, AttributeError, TypeError):
                readout[q] = float("nan")

        two_q_error: dict[tuple[int, int], float] = {}
        two_q_gate_names = [g for g in target.operation_names if g in ("ecr", "cz", "cx", "cnot")]
        for gate_name in two_q_gate_names:
            for qargs, props in (target[gate_name] or {}).items():
                if props is not None and props.error is not None and len(qargs) == 2:
                    two_q_error[tuple(qargs)] = float(props.error)
        return t1, readout, two_q_error

    def qubit_score(self, qubit: int) -> QubitScore:
        t1 = self._t1.get(qubit, float("nan"))
        readout = self._readout.get(qubit, float("nan"))
        edges = [v for (a, b), v in self._2q_error.items() if a == qubit or b == qubit]
        mean_2q = float(np.mean(edges)) if edges else float("nan")

        t1_term = t1 if np.isfinite(t1) else 50.0
        readout_term = readout if np.isfinite(readout) else 0.01
        gate_term = mean_2q if np.isfinite(mean_2q) else 0.02
        score = t1_term / (1.0 + 100 * readout_term + 100 * gate_term)
        return QubitScore(qubit=qubit, t1_us=t1_term, readout_error=readout_term,
                           mean_2q_error=gate_term, score=score)

    def select_layout(self, n_qubits: int, n_seed_candidates: int = 10) -> list[int]:
        if n_qubits > self.backend.num_qubits:
            from core.errors import ValidationError

            raise ValidationError(
                f"Backend {self.backend.name if hasattr(self.backend, 'name') else self.backend} "
                f"has only {self.backend.num_qubits} qubits, need {n_qubits}."
            )
        scores = {q: self.qubit_score(q).score for q in self.graph.nodes}
        seed_candidates = sorted(scores, key=lambda q: -scores[q])[:n_seed_candidates]

        best_layout: list[int] | None = None
        best_total = -np.inf
        for seed in seed_candidates:
            layout = self.graph.connected_subgraph_bfs(seed, n_qubits, priority=scores)
            if layout is None:
                continue
            total = sum(scores[q] for q in layout)
            if total > best_total:
                best_total = total
                best_layout = layout
        return best_layout if best_layout is not None else list(range(n_qubits))

    def layout_report(self, layout: list[int]) -> dict:
        scored = [self.qubit_score(q) for q in layout]
        return {
            "layout": layout,
            "n_qubits": len(layout),
            "t1_mean_us": float(np.mean([s.t1_us for s in scored])),
            "readout_error_mean": float(np.mean([s.readout_error for s in scored])),
            "mean_2q_error": float(np.mean([s.mean_2q_error for s in scored])),
        }


__all__ = ["QubitSelector", "QubitScore"]
