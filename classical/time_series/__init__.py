"""
time_series/ — Classical baselines for quantum component benchmarking.

MANDATORY: every quantum time-series/classification component in
algorithms/{qrc,vqc_qml,pqk} should be benchmarked against these before any
claim is made about it.

    classical_baselines.py  — LogisticBaseline (63.8% OOS accuracy is the
                              documented Level-1 go/no-go bar), LGBMBaseline,
                              RidgeBaseline. Requires scikit-learn (installed
                              in this venv); LGBMBaseline additionally requires
                              lightgbm (NOT installed here — gracefully
                              omitted from benchmark_all(), not a hard error).
    mamba_ssm_baseline.py   — Pure-numpy Selective State Space Model (Gu & Dao
                              2023, arXiv:2312.00752). No CUDA/mamba-ssm
                              package required for this implementation; it
                              trains only the linear head via SGD (SSM layers
                              use fixed random features) — a real, non-trivial
                              baseline with genuine input-dependent (selective)
                              dynamics, not merely "runs without erroring."

IMPORTANT — HISTORICAL CONTEXT, preserved verbatim from the source (do not
strip this when reusing these baselines): an internal Q10 audit (2026-07-11)
found PQK-v2, QRC-10, and VQC-v5 are classically simulable (CSIM). Beating
these baselines is NOT evidence of quantum advantage for those components —
it's a real benchmarking floor, not a marketing claim. The originally-planned
"Level 2" go/no-go bar (beat Mamba SSM) is WITHDRAWN for the same reason.
Quantum vs. Mamba SSM (2023) remains the scientifically honest comparison to
report, even though clearing it proves competence, not quantum advantage.

Reference: quantum_layer/study_notes/S9_bayesian_classical_ml.md (§6.4 Mamba)
"""
