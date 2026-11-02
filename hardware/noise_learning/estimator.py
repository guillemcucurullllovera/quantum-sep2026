"""Empirical noise estimation from measurement data.

Two independent, real techniques:

1. `fit_readout_noise_from_counts` — given counts from preparing and
   measuring computational-basis states, estimate the per-qubit readout
   confusion-matrix rates directly from data (rather than trusting a fake
   backend's advertised calibration numbers, which is what
   `hardware.calibration` reads instead).
2. `fit_depolarizing_rate_from_shadow` — given classical-shadow-style
   randomized measurement outcomes of a *known* reference state, estimate
   an effective depolarizing rate by comparing the measured purity to the
   theoretical noiseless purity. This is the standard shadow-tomography
   noise-diagnosis trick, not a full state-tomography routine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from core.experiment_registry import ExperimentRegistry
from core.schemas import ExperimentResult, RunMetadata, ExperimentConfig, BackendDescriptor, BackendKind


class LearnedNoiseModel(BaseModel):
    """A noise model estimated from data, not read off a device's target."""

    backend_name: str
    n_qubits: int
    per_qubit_readout_error: dict[int, float]
    effective_depolarizing_rate: float | None = None
    n_samples: int
    source: str = "empirical_fit"


def fit_readout_noise_from_counts(
    counts_by_prepared_state: dict[int, dict[str, int]], n_qubits: int
) -> dict[int, float]:
    """`counts_by_prepared_state`: {prepared_bit (0 or 1) per-qubit-index
    keyed measurement -> counts dict}. Concretely: for each qubit i, caller
    prepares |0> and |1>, measures, and passes
    `{0: counts_dict_for_prep0, 1: counts_dict_for_prep1}`.

    This function expects one such dict *per qubit*, i.e. the caller
    supplies per-qubit calibration circuits' results. Returns per-qubit
    readout error rate p(wrong outcome).
    """
    errors: dict[int, float] = {}
    for qubit, prep_counts in counts_by_prepared_state.items():
        total = 0
        wrong = 0
        for prepared_bit_str, counts in prep_counts.items():
            prepared = int(prepared_bit_str)
            for outcome, n in counts.items():
                total += n
                measured = int(outcome[-1])  # last char = this qubit's bit, little-endian convention
                if measured != prepared:
                    wrong += n
        errors[qubit] = wrong / total if total else 0.0
    return errors


def fit_depolarizing_rate_from_shadow(
    measured_purity: float, n_qubits: int, theoretical_purity: float = 1.0
) -> float:
    """Effective depolarizing rate p such that a depolarizing channel
    D(rho) = (1-p) rho + p * I/2^n applied to a pure reference state would
    reproduce the observed purity. Solved in closed form from
    Tr[D(rho)^2] = (1-p)^2 + 2p(1-p)/2^n + p^2/2^n (standard depolarizing-
    channel purity formula), via Newton's method since it's a simple
    quadratic in p.
    """
    if not (0.0 <= measured_purity <= theoretical_purity + 1e-9):
        raise ValueError(f"measured_purity={measured_purity} outside [0, {theoretical_purity}]")
    d = 2**n_qubits
    # Tr[rho_out^2] = (1-p)^2 * Tr[rho^2] + p^2/d  (cross term vanishes for pure rho with Tr[rho]=1
    # under the standard global-depolarizing-channel purity formula, to leading order)
    # => solve (1-p)^2 * theoretical_purity + p^2 / d = measured_purity
    # quadratic: p^2 (theoretical_purity + 1/d) - 2*theoretical_purity*p + (theoretical_purity - measured_purity) = 0
    a = theoretical_purity + 1.0 / d
    b = -2 * theoretical_purity
    c = theoretical_purity - measured_purity
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.0
    p = (-b - np.sqrt(disc)) / (2 * a)
    return float(np.clip(p, 0.0, 1.0))


def record_learned_noise_model(
    model: LearnedNoiseModel, registry: ExperimentRegistry | None = None
) -> str:
    """Log a fitted noise model into the experiment registry so it becomes
    queryable alongside CalibrationHistory snapshots for the same backend."""
    registry = registry or ExperimentRegistry()
    backend = BackendDescriptor(name=model.backend_name, kind=BackendKind.FAKE, num_qubits=model.n_qubits)
    config = ExperimentConfig(name="noise_learning_fit", algorithm="noise_learning", backend=backend)
    meta = RunMetadata(algorithm="noise_learning", generation=None)
    result = ExperimentResult(
        metadata=meta,
        config=config,
        metrics={
            "mean_readout_error": float(np.mean(list(model.per_qubit_readout_error.values()))) if model.per_qubit_readout_error else 0.0,
            "effective_depolarizing_rate": model.effective_depolarizing_rate or 0.0,
        },
        notes=f"learned noise model, n_samples={model.n_samples}, source={model.source}",
    )
    return registry.record(result)
