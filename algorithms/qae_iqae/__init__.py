"""Amplitude estimation for finance (QAE/IQAE) — Asymmetrica-original layer
built on top of algorithms_extended.qae_spinqit's ported circuit/estimator core.

This is the platform's one component with a genuine, literature-backed (if
modest and fault-tolerance-gated) quantum advantage claim — see
docs/QUANTUM_DEEP_REVISION.md. Everything else in algorithms/ is presented
as benchmarked research, not a claimed edge; keep that distinction honest
in anything built on top of this package.
"""
from algorithms.qae_iqae.register import register_qae_algorithm

register_qae_algorithm()
