"""
qrc/ — Quantum Reservoir Computing (QRC-10).

Architecture: 10q, IsingXX couplings, spectral radius α=0.9.
Readout: LGBM classifier on reservoir measurements.
Label: triple-barrier directional (buy/sell/hold) → C1.4 weight vector.
EM: DD (XY-4) + TREX only — decoherence is intentionally preserved.

CRITICAL BUG in phase0 _qrc.py (fixed here):
    random_couplings() was missing spectral radius normalization.
    Fix: J = J / np.max(np.abs(np.linalg.eigvalsh(J))) * alpha
    Without this, echo state property is violated and dynamics diverge.

Reference: quantum_layer/study_notes/S4_quantum_reservoir_computing.md
"""
