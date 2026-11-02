"""
feature_engineering/ — Financial ML feature engineering utilities.

All utilities are stateless and serializable (no side effects).
Required before any quantum OR classical model is trained on financial time series —
not optional hygiene.

    fracdiff.py        — Fractional differentiation (d≈0.35, López de Prado)
                         Use FFD (Fixed-Width Window) variant, not expanding.
                         Produces memory-preserving stationary series.
    purged_cv.py       — Purged k-fold CV (+ Combinatorial Purged K-Fold) with
                         embargo window. Prevents leakage from overlapping
                         return windows. See note below re: core.dataset_interfaces.
    triple_barrier.py  — Triple-barrier label generator (buy/sell/hold).
                         Originally required for QRC-10 and VQC v5 directional
                         training; equally applicable to any directional classifier
                         in algorithms/vqc_qml or algorithms/pqk.

Relationship to core.dataset_interfaces.FinancialDataset.split_walk_forward:
    split_walk_forward gives simple, non-overlapping, strictly-increasing
    walk-forward folds — no purging, no embargo, and it doesn't know about
    per-sample label *windows* (t1). PurgedKFold/CombinatorialPurgedKFold here
    are strictly more rigorous: they purge training rows whose label window
    overlaps a test fold's time span, and embargo rows immediately after it.
    Use split_walk_forward for a quick single-pass check with no label-window
    overlap risk (labels are single points in time); use PurgedKFold/CPCV
    whenever labels are computed over a forward-looking window (e.g. any
    triple_barrier_labels() output, whose t1 can extend well past t0) — that is
    exactly the scenario ordinary walk-forward splitting under-protects
    against. Both live under separate names on purpose; do not consolidate them
    into one function, they answer different validation questions.

Reference: quantum_layer/study_notes/S8_quantum_finance.md (§ Financial ML Hygiene)
"""
