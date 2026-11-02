"""Noise-aware QRC forecasting: run the canonical Gen2 persistent-state
architecture with and without an injected noise channel and report the
forecast-quality degradation honestly.

Modeling note (read before trusting these numbers as device-calibrated):
the QRC forward pass here is a direct density-matrix simulation (numpy
kron/einsum over the full evolution operator), not a gate-level Qiskit
circuit, so hardware.noise's Aer NoiseModel objects don't attach to it
directly the way they would to a transpiled circuit. Instead this module:

  1. Uses hardware.noise.models.synthetic()'s own parameterization
     (depolarizing_prob_1q, depolarizing_prob_2q, readout_error_prob) as the
     noise vocabulary, so the numbers are directly comparable to what a
     circuit-level Aer run would be configured with.
  2. Derives one effective per-timestep depolarizing probability by
     treating a single reservoir timestep as applying one 1-qubit error per
     qubit and one 2-qubit error per coupled pair in the Ising evolution
     (a Trotter-style accounting), then applies a *global* depolarizing
     channel rho -> (1-p) rho + p * I/dim to the joint state each step.
     This is a deliberate, documented approximation of gate-level noise --
     not a claim of full circuit-level fidelity.
  3. Applies symmetric readout error via the closed-form result for a
     Z-expectation under a bit-flip channel: <Z>_noisy = (1 - 2p) * <Z>_exact.
  4. Cross-checks the assumed noise level against
     hardware.noise_learning.estimator.fit_depolarizing_rate_from_shadow,
     which backs an effective depolarizing rate out of the *measured*
     purity of the noisy run's final joint state -- an independent,
     empirical consistency check on the same noise channel, not just the
     parameter we put in.
"""
from __future__ import annotations

import numpy as np

from algorithms.qrc.encoders import encode_sequence
from algorithms.qrc.reservoir import QuantumReservoir, QuantumReservoirConfig
from algorithms.qrc.reservoir_math import (
    encode_input_state,
    partial_trace_input,
    precompute_z_diagonals,
    ridge_fit,
    ridge_predict,
)
from applications.time_series.domain import MethodChoice, NoiseComparisonResult, NoiseSpec, TimeSeriesForecastSpec
from applications.time_series.errors import NoiseConfigError
from core.experiment_registry import ExperimentRegistry
from core.metrics import mae, mse
from hardware.noise.models import synthetic as synthetic_noise_model
from hardware.noise_learning.estimator import LearnedNoiseModel, fit_depolarizing_rate_from_shadow, record_learned_noise_model


def effective_depolarizing_prob(noise: NoiseSpec, n1: int, n2: int) -> float:
    """1 - (1-p1)^n * (1-p2)^(n choose 2) -- the Trotter-style approximation
    documented in this module's docstring."""
    n = n1 + n2
    n_pairs = n * (n - 1) // 2
    survive = (1 - noise.depolarizing_prob_1q) ** n * (1 - noise.depolarizing_prob_2q) ** n_pairs
    return float(np.clip(1.0 - survive, 0.0, 1.0))


def _noisy_forward(
    X: np.ndarray, U: np.ndarray, n1: int, n2: int, noise: NoiseSpec, washout: int = 0,
) -> tuple[np.ndarray, float]:
    """Persistent-state forward pass with a depolarizing channel applied to
    the joint state after every step's unitary evolution, and symmetric
    readout-error scaling applied to the extracted Z-expectation features.
    Returns (features, final_state_purity) -- the purity is used by
    `compare_noisy_vs_noiseless` as the hardware.noise_learning cross-check.
    """
    T, d = X.shape
    if d != n1:
        raise ValueError(f"X has {d} features but n1={n1}")
    n = n1 + n2
    dim = 1 << n
    dim_hidden = 1 << n2
    U_dag = U.conj().T
    z_diags = precompute_z_diagonals(n)
    p_eff = effective_depolarizing_prob(noise, n1, n2)
    readout_scale = 1.0 - 2.0 * noise.readout_error_prob
    identity_over_dim = np.eye(dim, dtype=np.complex128) / dim

    rho_hidden = np.zeros((dim_hidden, dim_hidden), dtype=np.complex128)
    rho_hidden[0, 0] = 1.0

    features = []
    rho = None
    for t in range(T):
        rho_input = encode_input_state(X[t])
        rho = np.kron(rho_input, rho_hidden)
        rho = U @ rho @ U_dag
        rho = (1 - p_eff) * rho + p_eff * identity_over_dim  # depolarizing channel
        rho_hidden = partial_trace_input(rho, n1, n2)
        if t >= washout:
            diag = np.real(np.diag(rho))
            features.append(readout_scale * (z_diags @ diag))

    final_purity = float(np.real(np.trace(rho @ rho))) if rho is not None else 1.0
    return np.array(features), final_purity


def noisy_qrc_sequence_features(X: np.ndarray, U: np.ndarray, n1: int, n2: int, noise: NoiseSpec, washout: int = 0) -> np.ndarray:
    """Public entry point returning just the noisy feature matrix (see
    `_noisy_forward` for the purity diagnostic used internally by
    `compare_noisy_vs_noiseless`)."""
    feats, _ = _noisy_forward(X, U, n1, n2, noise, washout)
    return feats


def compare_noisy_vs_noiseless(
    X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec, log_experiments: bool = True,
) -> NoiseComparisonResult:
    """Fit+evaluate the canonical Gen2 QRC method on one held-out split,
    once with the injected noise channel and once without, on the identical
    data/config -- the only difference is the noise. Uses a single split
    (not full walk-forward) to keep this a fast, targeted diagnostic rather
    than a duplicate of the main dispatcher's evaluation.
    """
    if not (0.0 <= spec.noise.readout_error_prob <= 0.5):
        raise NoiseConfigError(f"readout_error_prob={spec.noise.readout_error_prob} must be in [0, 0.5]")

    n = len(y)
    n_test = max(2, int(n * spec.test_fraction))
    if n - n_test < 5:
        raise NoiseConfigError(f"Series of length {n} is too short for a noise comparison split")
    train_idx = np.arange(0, n - n_test)
    test_idx = np.arange(n - n_test, n)

    cfg = QuantumReservoirConfig(
        n1=spec.qrc.n1, n2=spec.qrc.n2, alpha=spec.qrc.alpha, tau=spec.qrc.tau,
        field_nu=spec.qrc.field_nu, seed=spec.seed,
    )
    angles = encode_sequence(X, n1=cfg.n1)
    reservoir = QuantumReservoir(cfg).build()

    noiseless_feats = reservoir.sequence_features(angles, washout=0)
    noisy_feats, final_purity = _noisy_forward(angles, reservoir.U, cfg.n1, cfg.n2, spec.noise, washout=0)

    def _fit_eval(feats: np.ndarray) -> tuple[float, float]:
        tr = train_idx[train_idx < len(feats)]
        te = test_idx[test_idx < len(feats)]
        W = ridge_fit(feats[tr].T, y[tr], delta=spec.ridge_delta, bias=True)
        pred = ridge_predict(W, feats[te].T, bias=True)
        return mse(y[te], pred), mae(y[te], pred)

    noiseless_mse, noiseless_mae = _fit_eval(noiseless_feats)
    noisy_mse, noisy_mae = _fit_eval(noisy_feats)
    ratio = noisy_mse / noiseless_mse if noiseless_mse > 1e-12 else float("inf")

    n_total_qubits = cfg.n1 + cfg.n2
    empirical_rate = fit_depolarizing_rate_from_shadow(min(final_purity, 1.0), n_total_qubits, theoretical_purity=1.0)
    _, noise_descriptor = synthetic_noise_model(
        depolarizing_prob_1q=spec.noise.depolarizing_prob_1q,
        depolarizing_prob_2q=spec.noise.depolarizing_prob_2q,
        readout_error_prob=spec.noise.readout_error_prob,
    )

    if log_experiments:
        learned = LearnedNoiseModel(
            backend_name=noise_descriptor.backend_name,
            n_qubits=n_total_qubits,
            per_qubit_readout_error={i: spec.noise.readout_error_prob for i in range(n_total_qubits)},
            effective_depolarizing_rate=empirical_rate,
            n_samples=len(angles),
            source="qrc_noise_channel_purity_fit",
        )
        record_learned_noise_model(learned, ExperimentRegistry())

    notes = (
        f"assumed effective per-step depolarizing prob={effective_depolarizing_prob(spec.noise, cfg.n1, cfg.n2):.4f} "
        f"(Trotter-style approximation); readout scale={1 - 2 * spec.noise.readout_error_prob:.4f}; "
        f"hardware.noise_learning purity-fit cross-check gives an effective rate of {empirical_rate:.4f} "
        f"from final-state purity={final_purity:.4f} (noiseless purity is exactly 1.0 under unitary evolution)."
    )

    return NoiseComparisonResult(
        method=MethodChoice.QRC_GEN2,
        noiseless_mse=noiseless_mse, noisy_mse=noisy_mse, mse_degradation_ratio=ratio,
        noiseless_mae=noiseless_mae, noisy_mae=noisy_mae,
        noise_spec=spec.noise, notes=notes,
    )


__all__ = ["effective_depolarizing_prob", "noisy_qrc_sequence_features", "compare_noisy_vs_noiseless"]
