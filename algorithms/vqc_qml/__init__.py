"""Variational Quantum Classifier / QML algorithm family.

Two implementation generations:
- Gen1: circuit v1-v4, angle encoding,
    single-shot encoding, linear-chain entanglement, parameter-shift gradients.
    - Gen2: v5 — a hardware-efficient ansatz
    (HEA-S) with data re-uploading and five documented barren-plateau fixes
    (McClean 2018 qubit-count threshold, Cerezo 2021 near-identity init +
    layerwise pretraining, Jerbi 2023 / Schuld re-uploading richer Fourier
    spectrum, Schuld Representer Theorem ridge-regression readout).

Per the internal Q10 classical-simulability audit (2026-07-11), all VQC
variants built so far are classically simulable — beating a classical
baseline here is real, useful benchmarking evidence, not a quantum
advantage claim. See docs/research/Q10_AUDIT__classical-simulability.md
and docs/COMPLIANCE/NIST_AI_RMF_QUANTUM.md.
"""
from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from algorithms.vqc_qml.training import VQCTrainer, TrainingConfig

__all__ = ["VQCAnsatz", "VQCVersion", "VQCTrainer", "TrainingConfig"]
