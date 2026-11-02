"""Data-driven noise characterization, distinct from `hardware.noise`'s
a-priori/calibration-derived models: this module *fits* an empirical noise
model from measurement data (real or fake-backend-simulated), complementing
`hardware.calibration.CalibrationHistory`'s time-series of device snapshots
with a per-circuit statistical estimate of what the noise actually did.
"""
from hardware.noise_learning.estimator import (
    LearnedNoiseModel,
    fit_readout_noise_from_counts,
    fit_depolarizing_rate_from_shadow,
)

__all__ = ["LearnedNoiseModel", "fit_readout_noise_from_counts", "fit_depolarizing_rate_from_shadow"]
