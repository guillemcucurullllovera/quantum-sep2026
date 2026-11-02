"""Error mitigation techniques: DD (dynamical decoupling), PEC (probabilistic
error cancellation), TREX (twirled readout error expectation).

Readout-specific matrix-free mitigation lives in `hardware.readout_mitigation`
(M3/mthree, Apache-2.0). Zero-noise extrapolation lives in `hardware.zne`.

Note on scope: `algorithms_extended/` originally listed an Mitiq-based
error-mitigation adaptation as an example. Mitiq is GPL-3.0-licensed, and
this platform has deliberately avoided depending on or adapting GPL code
to keep the repository's licensing posture clean (documented previously
in an internal review that flagged Mitiq/QQuantLib as "reference-only,
not used"). Every technique in this package (and hardware.zne) is either
migrated Asymmetrica-internal code or implemented from the public
(non-copyrightable) method description, not ported from Mitiq's source.
"""
from hardware.mitigation.dd import DDMitigator
from hardware.mitigation.pec import PECMitigator
from hardware.mitigation.trex import TREXMitigator

__all__ = ["DDMitigator", "PECMitigator", "TREXMitigator"]
