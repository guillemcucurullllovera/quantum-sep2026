"""Multi-class market-regime detection, reusing the VQC classifier machinery
via a one-vs-rest RidgeReadout (already multiclass-capable — see
algorithms/vqc_qml/training.py:RidgeReadout).

This module adds `regime_product.py`: a confidence-routed ensemble of
`classical.meta_agent.RegimeHMM` and any supervised classifier reachable
through `applications.classification.dispatcher` -- "when to trust which,"
made measurable rather than left as a documented open question."""
