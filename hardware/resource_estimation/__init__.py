"""Campaign-level resource estimation, distinct from
`hardware.circuit_analysis` (which reports a single circuit's depth/gate
counts): this estimates the total cost — wall-clock time, shot budget,
queue-position risk — of running a full experiment campaign made of many
circuits, using per-circuit stats as an input, not a replacement.
"""
from hardware.resource_estimation.campaign import (
    CampaignEstimate,
    CircuitJob,
    estimate_campaign,
)

__all__ = ["CampaignEstimate", "CircuitJob", "estimate_campaign"]
