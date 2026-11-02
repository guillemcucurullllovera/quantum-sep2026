"""Tests for `experiments.quantum1b.quantum_lab.authorization_gate`.

Regression tests for independent-review findings P2-3 ("no code-level gate
tying real-service execution to the capstone's decision") and P2-4
("credentials_present is a trust-me boolean never derived from real
credential state")."""
from __future__ import annotations

import pytest

from experiments.quantum1b.quantum_lab.authorization_gate import (
    HardwareDayAuthorization,
    HardwareDayNotAuthorizedError,
    assert_hardware_day_authorized,
    check_hardware_day_authorization,
)


class TestCheckHardwareDayAuthorization:
    def test_returns_an_authorization_object(self):
        result = check_hardware_day_authorization()
        assert isinstance(result, HardwareDayAuthorization)

    def test_is_currently_not_authorized_in_this_checkout(self):
        """The load-bearing property this module exists for: given the
        real current repository state (no credentials, capstone
        NOT_READY, no independent-reviewer sign-off recorded), Hardware
        Day must NOT be authorized."""
        result = check_hardware_day_authorization()
        assert result.authorized is False

    def test_unmet_gates_includes_all_three_currently_closed_gates(self):
        result = check_hardware_day_authorization()
        assert "real_ibm_credentials_present" in result.unmet_gates
        assert "capstone_decision_is_hardware_day_ready" in result.unmet_gates
        assert "independent_review_signed_off" in result.unmet_gates

    def test_all_gates_dict_has_every_lab_wide_gate(self):
        result = check_hardware_day_authorization()
        assert "frozen_controls_verified" in result.all_gates
        assert "hardware_readiness_infra_tested" in result.all_gates


class TestAssertHardwareDayAuthorized:
    def test_raises_in_this_checkout(self):
        with pytest.raises(HardwareDayNotAuthorizedError):
            assert_hardware_day_authorized()

    def test_error_message_names_the_unmet_gates(self):
        with pytest.raises(HardwareDayNotAuthorizedError) as exc_info:
            assert_hardware_day_authorized()
        message = str(exc_info.value)
        assert "real_ibm_credentials_present" in message
        assert "capstone_decision_is_hardware_day_ready" in message


class TestCredentialsAreDerivedNotHandSupplied:
    """P2-4's specific concern: credentials_present must be DERIVED from
    real settings, never a caller-supplied literal this module trusts
    blindly."""

    def test_no_credentials_env_var_means_gate_closed(self, monkeypatch):
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        import core.config as config_mod

        config_mod.get_settings(refresh=True)
        try:
            result = check_hardware_day_authorization()
            assert result.all_gates["real_ibm_credentials_present"] is False
        finally:
            config_mod.get_settings(refresh=True)

    def test_setting_the_env_var_flips_only_the_credentials_gate(self, monkeypatch):
        """Even with a (dummy, non-functional) token present, authorization
        must still fail overall -- credentials alone are not sufficient,
        since the capstone decision and review sign-off gates are
        independently required."""
        monkeypatch.setenv("IBM_QUANTUM_TOKEN", "dummy-token-never-used-for-a-real-call")
        import core.config as config_mod

        config_mod.get_settings(refresh=True)
        try:
            result = check_hardware_day_authorization()
            assert result.all_gates["real_ibm_credentials_present"] is True
            assert result.authorized is False  # still blocked by the other gates
        finally:
            # **Bug found and fixed 2026-10-01** (discovered while auditing
            # the committed Command Center export for the Phase 5/
            # integration-handoff review -- the export was found to
            # incorrectly show `real_ibm_credentials_present: true` despite
            # no real token ever being configured). Root cause: calling
            # `get_settings(refresh=True)` here, BEFORE `monkeypatch`'s own
            # teardown has removed the env var set above (pytest fixture
            # teardown runs AFTER this test function returns, including
            # after this `finally` block), re-cached the DUMMY token's
            # "present" state into the GLOBAL `core.config` settings
            # singleton -- which then stayed polluted for the rest of the
            # pytest PROCESS (not just this test), silently corrupting any
            # later code in the same process (e.g. a test that regenerates
            # the Command Center export) that calls `get_settings()`
            # without its own explicit `refresh=True`. Fixed by explicitly
            # deleting the env var BEFORE refreshing, so the refresh
            # observes the real (no-token) state.
            monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
            config_mod.get_settings(refresh=True)
