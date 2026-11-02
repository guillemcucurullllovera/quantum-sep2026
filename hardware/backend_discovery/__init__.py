"""Backend discovery: "what backends can I even reach right now."

Distinct from `hardware.backend_selection`, which ranks backends already
known to be reachable — this module answers the prior question. Falls back
to the fake-backend catalog with an explicit `live=False` flag whenever no
IBM credentials are configured, so a caller never mistakes a fallback for a
real hardware inventory.
"""
from hardware.backend_discovery.discovery import DiscoveryResult, discover_backends

__all__ = ["DiscoveryResult", "discover_backends"]
