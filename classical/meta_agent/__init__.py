"""Classical statistical baselines directly comparable to this platform's
quantum counterparts, used to validate that a quantum approach is actually
earning its complexity rather than assumed better by default.

Pairings this enables:
  - RegimeHMM (classical, 3-state Gaussian HMM)  vs.  algorithms.vqc_qml /
    applications.regime_detection (quantum regime classification)
  - MonteCarloCVaR (classical, regime-conditional bootstrap)  vs.
    algorithms.qae_iqae / applications.tail_risk (QAE-based CVaR/VaR)
  - OCSVMAnomalyDetector -- general anomaly-detection baseline, complements
    the regime classifier.

Scope is intentionally narrow: only components with a direct quantum
counterpart worth benchmarking against are included here. Live
capital-allocation logic, backtesting harnesses, and operator tooling are
out of scope for this comparison layer.
"""
