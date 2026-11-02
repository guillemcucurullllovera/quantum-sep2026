"""Registers the qae_iqae algorithm with core's plugin registry and provides the
clean, application-facing API applications/{options,derivatives,tail_risk} call into.
"""
from __future__ import annotations

import time

from algorithms.qae_iqae.finance_integrands import BarrierIntegrand, build_crash_barrier
from algorithms.qae_iqae.iqae import run_barrier_demo
from core.experiment_registry import ExperimentRegistry
from core.registry import ALGORITHM_REGISTRY
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    MitigationConfig,
    MitigationKind,
    RunMetadata,
)


def estimate_barrier_probability(
    barrier: BarrierIntegrand,
    *,
    epsilon: float = 0.01,
    alpha: float = 0.05,
    shots: int = 2048,
    seed: int = 11,
    mitigate: bool = True,
    log_to_registry: bool = True,
) -> ExperimentResult:
    """The stable, application-facing entry point: estimate P(event) for a barrier-style
    integrand via IQAE (ideal + noisy simulator), and return a typed ExperimentResult
    with both the quantum estimate and enough metadata to compare against a classical
    baseline computed by the caller (applications.tail_risk / applications.options do
    this, so the comparison always uses the platform's shared metric definitions)."""
    t0 = time.time()
    raw = run_barrier_demo(barrier, epsilon=epsilon, alpha=alpha, shots=shots, seed=seed, m3=mitigate)
    wall_time = time.time() - t0

    backend = BackendDescriptor(
        name=raw["config"]["backend"], kind=BackendKind.FAKE,
        num_qubits=raw["config"]["backend_num_qubits"],
        calibration_source=f"fake_backend:{raw['config']['backend']}",
    )
    mitigation = MitigationConfig(kind=MitigationKind.READOUT_M3 if mitigate else MitigationKind.NONE)
    config = ExperimentConfig(
        name=f"qae_iqae::{barrier.name}", algorithm="qae_iqae", backend=backend,
        mitigation=mitigation, shots=shots, seed=seed,
        params={"epsilon": epsilon, "alpha": alpha, "barrier_label": barrier.label},
    )
    metadata = RunMetadata(algorithm="qae_iqae", generation="gen2", adapted_from="spinqit-qae-triangulum (MIT)")

    metrics = {
        "a_hat_ideal": raw["modes"]["ideal"]["a_hat"],
        "a_hat_noisy": raw["modes"]["kingston_noisy"]["a_hat"],
        "exact_grid_a": raw["exact_grid_a"],
        "exact_analytic": raw["exact_analytic"],
        "abs_error_ideal_vs_grid": raw["modes"]["ideal"]["abs_error_vs_grid"],
        "abs_error_noisy_vs_grid": raw["modes"]["kingston_noisy"]["abs_error_vs_grid"],
        "discretisation_gap": raw["discretisation_gap"] or 0.0,
        "wall_time_s": wall_time,
        "circuit_depth_Q2A": raw["circuits"].get("Q2A", {}).get("depth", 0),
        "two_qubit_gates_Q2A": raw["circuits"].get("Q2A", {}).get("twoq_gates", 0),
    }
    if raw.get("m3"):
        metrics["p1_raw"] = raw["m3"]["p1_raw"]
        metrics["p1_mitigated"] = raw["m3"]["p1_mitigated"]

    result = ExperimentResult(
        metadata=metadata, config=config, metrics=metrics,
        notes=raw["provenance"]["note"],
    )
    if log_to_registry:
        ExperimentRegistry().record(result)
    return result


def register_qae_algorithm() -> None:
    if "qae_iqae" in ALGORITHM_REGISTRY.list():
        return
    ALGORITHM_REGISTRY.register("qae_iqae")(estimate_barrier_probability)
