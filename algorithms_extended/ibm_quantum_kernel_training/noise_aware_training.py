"""Noise-aware extension: train a covariant feature map's KTA under a
simulated noisy backend rather than a noiseless simulator.

Neither the upstream toolkit nor `kta_alignment.py` trains under
noise -- both assume a noiseless statevector/shot simulator. This is a real
gap: a kernel trained to be well-aligned in the noiseless case is not
guaranteed to stay well-aligned once shot noise and gate/readout error are
present, which matters for anything actually destined for hardware. This
module is genuinely new (no upstream equivalent), built on `hardware.noise` infrastructure rather than reimplementing noise modelling.
"""
from __future__ import annotations

import numpy as np
from qiskit_aer import AerSimulator

from algorithms_extended.ibm_quantum_kernel_training.qkt_trainer import spsa_kta_train
from hardware.noise import models as noise_models


def train_under_noise(
    feature_map,
    X: np.ndarray,
    y: np.ndarray,
    theta_init: np.ndarray,
    depolarizing_prob_1q: float = 0.001,
    depolarizing_prob_2q: float = 0.01,
    readout_error_prob: float = 0.02,
    maxiter: int = 60,
    shots: int = 512,
) -> dict:
    """Train once noiselessly and once under a synthetic noise model (Wave
    1's `hardware.noise.models.synthetic`), on identical data/init, and
    report both -- an honest before/after, not a single noisy-only number.
    """
    noise_model, descriptor = noise_models.synthetic(
        depolarizing_prob_1q=depolarizing_prob_1q,
        depolarizing_prob_2q=depolarizing_prob_2q,
        readout_error_prob=readout_error_prob,
    )
    backend = AerSimulator()

    clean_theta, clean_kta, clean_hist = spsa_kta_train(
        feature_map, X, y, theta_init, maxiter=maxiter, backend=backend, noise_model=None, shots=shots
    )
    noisy_theta, noisy_kta, noisy_hist = spsa_kta_train(
        feature_map, X, y, theta_init, maxiter=maxiter, backend=backend, noise_model=noise_model, shots=shots
    )

    return {
        "noise_descriptor": descriptor.model_dump(),
        "noiseless": {"best_kta": clean_kta, "history": clean_hist, "theta": clean_theta.tolist()},
        "noisy": {"best_kta": noisy_kta, "history": noisy_hist, "theta": noisy_theta.tolist()},
        "kta_degradation": clean_kta - noisy_kta,
    }
