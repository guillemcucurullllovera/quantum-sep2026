"""Gen2 QAOA portfolio implementation.

This is the
recommended default engine — see ../README.md for why it supersedes gen1's
approximation-ratio performance.
"""
from algorithms.qaoa_qubo.gen2.qaoa_v2 import QAOAv2, QAOAConfig
from algorithms.qaoa_qubo.gen2.mixers import (
    ring_xy_pairs,
    correlation_xy_pairs,
    complete_xy_pairs,
    apply_mixer,
    initial_dicke_state,
    plus_state,
)
from algorithms.qaoa_qubo.gen2.sk_warmstart import (
    sk_warmstart_angles,
    sk_warmstart_from_qubo,
    random_angles,
    compute_j_scale,
)

__all__ = [
    "QAOAv2", "QAOAConfig",
    "ring_xy_pairs", "correlation_xy_pairs", "complete_xy_pairs",
    "apply_mixer", "initial_dicke_state", "plus_state",
    "sk_warmstart_angles", "sk_warmstart_from_qubo", "random_angles", "compute_j_scale",
]
