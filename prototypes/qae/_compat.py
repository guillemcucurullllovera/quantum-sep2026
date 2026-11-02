"""Shared compatibility shim for the ported QAE prototype CLIs (`run_sim.py`, `run_ibm.py`).

The original `quantum/qae/integrands.py:get_integrand` was split across
`algorithms_extended.qae_spinqit.toy_integrands` (the TOY registry) and
`algorithms.qae_iqae.finance_integrands` (the finance integrand) by port —
this re-joins them for these CLIs' lookup-by-name convenience, unchanged in behavior
from the original.
"""
from __future__ import annotations

from algorithms.qae_iqae.finance_integrands import build_call_spread_integrand
from algorithms_extended.qae_spinqit.toy_integrands import TOY, Integrand


def get_integrand(name: str) -> Integrand:
    if name in TOY:
        return TOY[name]
    if name in ("finance", "finance_call_spread", "call_spread"):
        return build_call_spread_integrand()
    raise KeyError(f"unknown integrand {name!r}; toy={list(TOY)} or 'finance_call_spread'")
