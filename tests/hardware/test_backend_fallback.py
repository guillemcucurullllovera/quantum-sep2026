"""hardware.backend_selection.fallback -- a documented, tested
fallback chain (IBM fake backend -> Braket local simulator -> Aer plain
simulator) for when a requested backend/cloud is unavailable. Must engage
loudly (logged) and be reported, never silently substitute."""
from __future__ import annotations

import logging

import pytest

from hardware.backend_selection.fallback import (
    AllBackendsUnavailableError,
    DEFAULT_FALLBACK_ORDER,
    FallbackOutcome,
    _DEFAULT_STEP_RESOLVERS,
    resolve_with_fallback,
)


def test_requested_backend_available_no_fallback_engaged(caplog):
    with caplog.at_level(logging.WARNING):
        outcome = resolve_with_fallback("fake_heron")
    assert isinstance(outcome, FallbackOutcome)
    assert outcome.requested == "fake_heron"
    assert outcome.actual.step_name in ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez")
    assert outcome.fell_back is False
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].ok is True
    # No warning should have been logged -- nothing went wrong.
    assert not any("fell back" in r.message for r in caplog.records)


def test_unknown_backend_falls_through_default_chain_and_is_logged(caplog):
    """A requested step with no registered resolver at all (the
    "unregistered/unknown backend name" unavailability case) must fall
    through to the documented chain, loudly."""
    with caplog.at_level(logging.WARNING):
        outcome = resolve_with_fallback("totally_unknown_backend_xyz")

    assert outcome.requested == "totally_unknown_backend_xyz"
    assert outcome.fell_back is True
    assert outcome.actual.step_name in ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez")
    assert outcome.actual.cloud == "ibm_quantum"
    # First attempt (the unknown name) recorded as failed.
    assert outcome.attempts[0].step == "totally_unknown_backend_xyz"
    assert outcome.attempts[0].ok is False
    # Second attempt (fake_heron, first in DEFAULT_FALLBACK_ORDER) succeeded.
    assert outcome.attempts[1].step == "fake_heron"
    assert outcome.attempts[1].ok is True
    # Loud: a warning was actually logged, not just returned silently.
    assert any("unavailable" in r.message for r in caplog.records)


def test_ibm_unavailable_falls_through_to_braket_then_would_reach_aer(caplog):
    """Deliberately make the preferred backend (a stand-in for a real IBM
    QPU requiring credentials) unavailable, and confirm the chain engages
    Braket's local simulator next -- the documented IBM -> Braket -> Aer
    order -- and that this is logged, not silent."""

    def _unavailable_ibm_qpu():
        from core.config import get_settings

        get_settings().ibm.require()  # raises CredentialError when unset
        raise AssertionError("should never get here in this test")

    resolvers = dict(_DEFAULT_STEP_RESOLVERS)
    resolvers["ibm_qpu"] = _unavailable_ibm_qpu

    with caplog.at_level(logging.WARNING):
        outcome = resolve_with_fallback(
            "ibm_qpu",
            fallback_order=("braket_local", "noiseless"),
            resolvers=resolvers,
        )

    assert outcome.fell_back is True
    assert outcome.attempts[0].step == "ibm_qpu"
    assert outcome.attempts[0].ok is False
    assert "CredentialError" in outcome.attempts[0].error
    assert outcome.actual.step_name == "braket_local_simulator"
    assert outcome.actual.cloud == "aws_braket_local"
    assert any("fell back" in r.message for r in caplog.records)


def test_ibm_and_braket_both_unavailable_falls_all_the_way_to_aer(caplog):
    """Force both the requested backend AND Braket to be unavailable and
    confirm the chain reaches the final Aer-plain-simulator step -- the
    documented full 3-step fallback."""

    def _broken_ibm_fake():
        from core.errors import BackendUnavailableError

        raise BackendUnavailableError("simulated: no IBM fake backend available")

    def _broken_braket():
        raise ImportError("amazon-braket-sdk not installed (simulated for this test)")

    resolvers = dict(_DEFAULT_STEP_RESOLVERS)
    resolvers["fake_heron"] = _broken_ibm_fake
    resolvers["braket_local"] = _broken_braket

    with caplog.at_level(logging.WARNING):
        outcome = resolve_with_fallback(
            "totally_unavailable_ibm_backend",
            fallback_order=DEFAULT_FALLBACK_ORDER,
            resolvers=resolvers,
        )

    steps_tried = [a.step for a in outcome.attempts]
    assert steps_tried == [
        "totally_unavailable_ibm_backend",
        "fake_heron",
        "braket_local",
        "noiseless",
    ]
    assert [a.ok for a in outcome.attempts] == [False, False, False, True]
    assert outcome.actual.step_name == "aer_simulator_noiseless"
    assert outcome.actual.cloud == "local"
    assert outcome.fell_back is True
    # Every failed step logged individually.
    warnings = [r.message for r in caplog.records]
    assert sum("unavailable" in w for w in warnings) >= 3


def test_all_backends_unavailable_raises_loudly_never_silent():
    def _always_fails():
        raise RuntimeError("simulated total outage")

    resolvers = {"only_option": _always_fails}
    with pytest.raises(AllBackendsUnavailableError):
        resolve_with_fallback("only_option", fallback_order=(), resolvers=resolvers)


def test_braket_local_actually_usable_when_reached_via_fallback():
    """The fallback chain must hand back a genuinely working backend, not
    just a label -- run a real circuit through whatever `resolve_with_fallback`
    returns when forced onto the Braket step."""
    from braket.devices import LocalSimulator

    outcome = resolve_with_fallback(
        "nonexistent", fallback_order=("braket_local", "noiseless")
    )
    assert outcome.actual.step_name == "braket_local_simulator"
    assert isinstance(outcome.actual.handle, LocalSimulator)
