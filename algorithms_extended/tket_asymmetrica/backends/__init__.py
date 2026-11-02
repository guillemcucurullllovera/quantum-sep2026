"""Adapted from pytket-quantinuum (Apache-2.0). See THIRD_PARTY_NOTICES.md.

Only the hardware-agnostic utility modules are included here (auth-config
schema, leakage-detection gadget) — the actual Quantinuum cloud API-wrapper
and job-submission backend classes were deliberately NOT copied, since they
require a live Quantinuum account/token and would be dead, untestable code
in this environment. See THIRD_PARTY_NOTICES.md for the full boundary.
"""
from algorithms_extended.tket_asymmetrica.backends.config import QuantinuumConfig
from algorithms_extended.tket_asymmetrica.backends.leakage_gadget import (
    get_detection_circuit,
    get_leakage_gadget_circuit,
    prune_shots_detected_as_leaky,
)

__all__ = [
    "QuantinuumConfig",
    "get_detection_circuit",
    "get_leakage_gadget_circuit",
    "prune_shots_detected_as_leaky",
]
