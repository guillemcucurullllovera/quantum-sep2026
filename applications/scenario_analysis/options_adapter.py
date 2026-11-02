"""Composition-point note, not a reimplementation: options-specific spot/vol
scenario sweeps already live in
`applications/derivatives/scenario_analysis.py` (territory) and
`algorithms_extended/qiskit_finance_extensions/scenario_analysis.py`
(barrier-specific). This module does not duplicate either -- it re-exports
`run_spot_vol_sweep` under this package's namespace so a caller doing
cross-application scenario work (tail-risk + options together) has one
place to discover both, matching how `applications/scenario_analysis`'s
`__init__.py` already re-exported the derivatives sweep previously.
"""
from __future__ import annotations

from applications.derivatives.scenario_analysis import ScenarioSweepResult, run_spot_vol_sweep

__all__ = ["ScenarioSweepResult", "run_spot_vol_sweep"]
