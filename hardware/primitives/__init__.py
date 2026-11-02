"""Alias package: `hardware.primitives` re-exports `hardware.ibm_runtime`'s
Sampler/Estimator wrappers under the shorter, primitive-focused name the
target architecture calls for, without duplicating the implementation."""
from hardware.ibm_runtime.primitives import EstimatorRunResult, SamplerRunResult, run_estimator, run_sampler

__all__ = ["run_sampler", "run_estimator", "SamplerRunResult", "EstimatorRunResult"]
