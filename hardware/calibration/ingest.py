"""Extract and normalize calibration data (T1/T2, readout error, two-qubit
gate error) directly from a Qiskit `BackendV2.target` — real numbers, never
fabricated placeholders when a value is missing.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class CalibrationSnapshot(BaseModel):
    backend_name: str
    n_qubits: int
    captured_at: float = Field(default_factory=time.time)
    t1_us: dict[int, float] = Field(default_factory=dict)
    t2_us: dict[int, float] = Field(default_factory=dict)
    readout_error: dict[int, float] = Field(default_factory=dict)
    two_qubit_error_mean: float | None = None
    two_qubit_error_samples: int = 0
    t1_mean_us: float | None = None


def extract_calibration(backend) -> CalibrationSnapshot:
    target = backend.target
    t1: dict[int, float] = {}
    t2: dict[int, float] = {}
    readout: dict[int, float] = {}

    if target.qubit_properties:
        for q, qp in enumerate(target.qubit_properties):
            if qp is None:
                continue
            if qp.t1 is not None:
                t1[q] = qp.t1 * 1e6  # seconds -> microseconds
            if qp.t2 is not None:
                t2[q] = qp.t2 * 1e6

    two_q_errors: list[float] = []
    for inst_name in target.operation_names:
        try:
            props_map = target[inst_name]
        except Exception:
            continue
        if not isinstance(props_map, dict):
            continue
        for qargs, props in props_map.items():
            if props is None or props.error is None or qargs is None:
                continue
            if len(qargs) == 2:
                two_q_errors.append(props.error)
            elif len(qargs) == 1 and inst_name.lower() == "measure":
                readout[qargs[0]] = props.error

    return CalibrationSnapshot(
        backend_name=backend.name,
        n_qubits=backend.num_qubits,
        t1_us=t1,
        t2_us=t2,
        readout_error=readout,
        two_qubit_error_mean=statistics.mean(two_q_errors) if two_q_errors else None,
        two_qubit_error_samples=len(two_q_errors),
        t1_mean_us=statistics.mean(t1.values()) if t1 else None,
    )
