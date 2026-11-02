"""Tests for `experiments.quantum1b.quantum_lab.readiness_state`."""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.quantum1b.quantum_lab.readiness_state import (
    CAPSTONE_DECISION,
    STATE_ORDER,
    INDEPENDENT_REVIEW_SIGNED_OFF,
    ReadinessGates,
    ReadinessState,
    compute_readiness_state,
    current_lab_wide_gates,
)


def _base_gates(**overrides) -> ReadinessGates:
    defaults = dict(
        has_classical_baseline=True,
        has_ideal_simulator_result=False,
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=False,
        frozen_controls_verified=False,
        hardware_readiness_infra_tested=False,
        independent_review_signed_off=False,
        real_ibm_credentials_present=False,
        capstone_decision_is_hardware_day_ready=False,
        verified_hardware_attestation_exists=False,
    )
    defaults.update(overrides)
    return ReadinessGates(**defaults)


class TestComputeReadinessState:
    def test_no_classical_baseline_raises(self):
        with pytest.raises(ValueError):
            compute_readiness_state(_base_gates(has_classical_baseline=False))

    def test_floor_state_is_research(self):
        assert compute_readiness_state(_base_gates()) == ReadinessState.RESEARCH

    def test_ideal_simulator_result_alone_gives_simulator_validated(self):
        gates = _base_gates(has_ideal_simulator_result=True)
        assert compute_readiness_state(gates) == ReadinessState.SIMULATOR_VALIDATED

    def test_noise_validated_result_alone_gives_noise_validated(self):
        # Even without an ideal-simulator flag set, a noise-validated result
        # is itself evidence of a working simulator pipeline -- it is a
        # strictly later stage, not an alternative to SIMULATOR_VALIDATED.
        gates = _base_gates(has_noise_validated_result=True)
        assert compute_readiness_state(gates) == ReadinessState.NOISE_VALIDATED

    def test_noise_validated_without_hardware_pathway_does_not_advance_further(self):
        gates = _base_gates(
            has_noise_validated_result=True,
            frozen_controls_verified=True,
            hardware_readiness_infra_tested=True,
            hardware_readiness_pathway_exists=False,  # e.g. Track C/E
        )
        assert compute_readiness_state(gates) == ReadinessState.NOISE_VALIDATED

    def test_full_infra_plus_pathway_reaches_hardware_ready_for_review(self):
        gates = _base_gates(
            has_noise_validated_result=True,
            hardware_readiness_pathway_exists=True,
            frozen_controls_verified=True,
            hardware_readiness_infra_tested=True,
        )
        assert compute_readiness_state(gates) == ReadinessState.HARDWARE_READY_FOR_REVIEW

    def test_missing_frozen_controls_verification_blocks_hardware_ready_for_review(self):
        gates = _base_gates(
            has_noise_validated_result=True,
            hardware_readiness_pathway_exists=True,
            frozen_controls_verified=False,
            hardware_readiness_infra_tested=True,
        )
        assert compute_readiness_state(gates) == ReadinessState.NOISE_VALIDATED

    def test_hardware_ready_requires_credentials_capstone_and_review_signoff(self):
        base = dict(
            has_noise_validated_result=True,
            hardware_readiness_pathway_exists=True,
            frozen_controls_verified=True,
            hardware_readiness_infra_tested=True,
        )
        # Missing credentials -> capped at HARDWARE_READY_FOR_REVIEW.
        gates = _base_gates(
            **base, capstone_decision_is_hardware_day_ready=True, independent_review_signed_off=True,
            real_ibm_credentials_present=False,
        )
        assert compute_readiness_state(gates) == ReadinessState.HARDWARE_READY_FOR_REVIEW

        # All four present -> HARDWARE_READY.
        gates = _base_gates(
            **base, capstone_decision_is_hardware_day_ready=True, independent_review_signed_off=True,
            real_ibm_credentials_present=True,
        )
        assert compute_readiness_state(gates) == ReadinessState.HARDWARE_READY

    def test_hardware_executed_requires_verified_attestation_plus_hardware_ready_gates(self):
        gates = _base_gates(
            has_noise_validated_result=True,
            hardware_readiness_pathway_exists=True,
            frozen_controls_verified=True,
            hardware_readiness_infra_tested=True,
            capstone_decision_is_hardware_day_ready=True,
            independent_review_signed_off=True,
            real_ibm_credentials_present=True,
            verified_hardware_attestation_exists=True,
        )
        assert compute_readiness_state(gates) == ReadinessState.HARDWARE_EXECUTED

    def test_attestation_alone_without_credentials_or_capstone_does_not_reach_executed(self):
        gates = _base_gates(
            has_noise_validated_result=True,
            hardware_readiness_pathway_exists=True,
            frozen_controls_verified=True,
            hardware_readiness_infra_tested=True,
            verified_hardware_attestation_exists=True,
            # credentials/capstone/review all still False
        )
        assert compute_readiness_state(gates) != ReadinessState.HARDWARE_EXECUTED

    def test_state_order_is_consistent_with_the_enum(self):
        assert set(STATE_ORDER) == set(ReadinessState)
        assert len(STATE_ORDER) == len(set(STATE_ORDER))


class TestCurrentLabWideGatesReflectRealRepoState:
    """These tests make real, local, network-free calls -- `is_present` only
    checks an environment variable, `verify_frozen_controls()` only touches
    local code and the committed snapshot file. No IBM credentials, no
    network call, anywhere in this test class."""

    def test_returns_all_expected_keys(self):
        gates = current_lab_wide_gates()
        expected_keys = {
            "frozen_controls_verified", "hardware_readiness_infra_tested",
            "independent_review_signed_off", "real_ibm_credentials_present",
            "capstone_decision_is_hardware_day_ready", "verified_hardware_attestation_exists",
        }
        assert set(gates) == expected_keys
        assert all(isinstance(v, bool) for v in gates.values())

    def test_frozen_controls_verified_is_true_in_a_clean_checkout(self):
        """Regression guard: if this goes False unexpectedly, either the
        committed snapshot or the generator code has drifted -- see P1-5."""
        gates = current_lab_wide_gates()
        assert gates["frozen_controls_verified"] is True

    def test_no_real_credentials_in_this_test_environment(self, monkeypatch):
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        import core.config as config_mod

        config_mod.get_settings(refresh=True)
        try:
            gates = current_lab_wide_gates()
            assert gates["real_ibm_credentials_present"] is False
        finally:
            config_mod.get_settings(refresh=True)

    def test_immune_to_a_stale_polluted_global_settings_cache(self, monkeypatch):
        """Regression guard for a real bug found and fixed 2026-10-01: a
        committed Command Center export incorrectly showed
        `real_ibm_credentials_present: true` because some OTHER test
        earlier in the same pytest process left the process-global
        `core.config` settings singleton cached with a dummy credential's
        "present" state (a monkeypatch-teardown-ordering bug in two test
        files, both now fixed separately). This test proves
        `current_lab_wide_gates()` itself no longer depends on every other
        test in the suite practicing perfect hygiene: it deliberately
        pollutes the cache exactly the way the original bug did, WITHOUT
        cleaning it up first, and confirms `current_lab_wide_gates()`
        still returns the correct (no-credentials) answer anyway, because
        it forces its own fresh read via `get_settings(refresh=True)`."""
        import core.config as config_mod

        # Simulate the exact pollution pattern: set a dummy token, refresh
        # (caching "present"), then remove the env var WITHOUT refreshing
        # again -- reproducing the stale-cache state the original bug left
        # behind, deliberately, as the starting condition for this test.
        monkeypatch.setenv("IBM_QUANTUM_TOKEN", "dummy-token-simulating-a-polluted-cache")
        config_mod.get_settings(refresh=True)
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        # Deliberately do NOT call get_settings(refresh=True) here -- the
        # whole point is to leave the cache polluted and prove the gate
        # function recovers on its own.
        assert config_mod.get_settings().ibm.is_present is True, (
            "test setup assumption failed: the cache should be polluted at this point"
        )

        try:
            gates = current_lab_wide_gates()
            assert gates["real_ibm_credentials_present"] is False, (
                "current_lab_wide_gates() returned a STALE cached credential state instead of "
                "refreshing -- this is exactly the bug that corrupted a committed export"
            )
        finally:
            config_mod.get_settings(refresh=True)

    def test_hardware_day_gates_all_currently_closed(self):
        """The whole point of this module: nothing reaches HARDWARE_READY or
        HARDWARE_EXECUTED right now. This test fails loudly the instant that
        stops being true, which is exactly the point -- those states must
        never flip open silently."""
        gates = current_lab_wide_gates()
        assert gates["independent_review_signed_off"] is False
        assert gates["capstone_decision_is_hardware_day_ready"] is False
        assert gates["verified_hardware_attestation_exists"] is False


class TestCapstoneDecisionConstantMatchesTheDocument:
    def test_capstone_decision_constant_matches_the_document(self):
        """P1-5-style anti-drift guard: CAPSTONE_DECISION is a hand-set
        constant (not parsed from markdown at runtime, deliberately -- see
        the module docstring), so this test is what keeps it honest. If the
        capstone document's decision changes, this test fails until the
        constant is updated to match -- it cannot silently drift."""
        capstone_path = (
            Path(__file__).parents[3] / "docs" / "quantum" / "PRE_HARDWARE_QUANTUM_CAPSTONE.md"
        )
        text = capstone_path.read_text()
        assert f"Decision: `{CAPSTONE_DECISION}`" in text, (
            f"CAPSTONE_DECISION={CAPSTONE_DECISION!r} does not match the capstone "
            "document's actual stated decision -- update the constant in "
            "readiness_state.py to match."
        )

    def test_independent_review_signed_off_constant_is_false(self):
        """This constant must be flipped to True only by a human recording
        an actual sign-off -- it must never default to True."""
        assert INDEPENDENT_REVIEW_SIGNED_OFF is False
