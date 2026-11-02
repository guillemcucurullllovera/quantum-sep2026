"""Tests for `experiments.quantum1b.hardware_readiness.frozen_controls`."""
from __future__ import annotations

import dataclasses

from experiments.quantum1b.hardware_readiness import frozen_controls as fc_mod
from experiments.quantum1b.hardware_readiness.frozen_controls import (
    FROZEN_CONTROL_SET,
    FROZEN_CONTROLS,
    load_committed_snapshot,
    verify_frozen_controls,
    write_snapshot,
)


class TestFrozenControls:
    def test_frozen_controls_matches_control_set_length(self):
        assert len(FROZEN_CONTROLS) == len(FROZEN_CONTROL_SET)

    def test_every_control_has_nonempty_fingerprint(self):
        for c in FROZEN_CONTROLS:
            assert c.expected_fingerprint
            assert len(c.expected_fingerprint) >= 32

    def test_every_control_has_positive_qubit_count(self):
        for c in FROZEN_CONTROLS:
            assert c.expected_n_qubits > 0

    def test_verify_frozen_controls_passes_against_the_committed_snapshot(self):
        """`verify_frozen_controls()`'s default argument now loads
        `frozen_controls_snapshot.json` from disk (fixed 2026-10-01,
        remediating P1-5), not `FROZEN_CONTROLS` -- this test confirms the
        committed snapshot is currently in sync with what the code
        regenerates (i.e. nobody has changed generators.py/qubo_encoding.py/
        classical_exact.py without regenerating and committing a fresh
        snapshot)."""
        results = verify_frozen_controls()
        assert len(results) == len(FROZEN_CONTROLS)
        for r in results:
            assert r.all_match, r

    def test_verify_frozen_controls_detects_a_deliberately_wrong_expectation(self):
        tampered = dataclasses.replace(FROZEN_CONTROLS[0], expected_fingerprint="deliberately-wrong")
        results = verify_frozen_controls((tampered,) + FROZEN_CONTROLS[1:])
        assert results[0].fingerprint_matches is False
        assert results[0].all_match is False
        for r in results[1:]:
            assert r.all_match


class TestSnapshotFile:
    """Regression tests for independent-review finding P1-5 ('"Frozen"
    classical controls are not actually frozen; the drift-detection
    mechanism cannot catch the failure mode it is named for'). Before the
    fix, `verify_frozen_controls()` defaulted to comparing a fresh
    recomputation against `FROZEN_CONTROLS` -- itself a fresh recomputation
    from the SAME import of the SAME code -- so the check could never catch
    a change to `generators.py`/`qubo_encoding.py`/`classical_exact.py`
    between when the frozen set was chosen and when the check runs. These
    tests reproduce the review's exact scenario: a semantic change to
    `generate_instance` now DOES produce a mismatch, because the comparison
    target is a committed file, independent of the current import."""

    def test_round_trip_through_a_snapshot_file_is_lossless(self, tmp_path):
        path = tmp_path / "snap.json"
        write_snapshot(FROZEN_CONTROLS, path)
        loaded = load_committed_snapshot(path)
        assert loaded == FROZEN_CONTROLS

    def test_default_argument_loads_from_the_real_committed_file_on_disk(self):
        """Confirms the committed `frozen_controls_snapshot.json` this
        session generated and committed is the actual file
        `verify_frozen_controls()`'s default reads -- not a copy, not
        `FROZEN_CONTROLS` re-serialized on the fly."""
        from_default = verify_frozen_controls()
        from_explicit_file_load = verify_frozen_controls(load_committed_snapshot(fc_mod._SNAPSHOT_PATH))
        assert from_default == from_explicit_file_load

    def test_a_semantic_change_to_generate_instance_is_now_caught_against_the_unchanged_snapshot(
        self, monkeypatch
    ):
        """The exact failure mode P1-5 named: this reproduces 'make a small
        semantic change to generators.py, re-run verify_frozen_controls(),
        confirm it now reports a mismatch' (the review's own specified
        reproduction/verification step) WITHOUT actually editing
        generators.py -- monkeypatching `frozen_controls.generate_instance`
        stands in for a real code change reaching the same call site, and
        the committed snapshot file on disk is left completely untouched,
        exactly mirroring what a real drift between 'when the controls were
        frozen' and 'Hardware Day' would look like."""
        real_generate_instance = fc_mod.generate_instance

        def drifted_generate_instance(family, n_candidates, seed=0):
            inst = real_generate_instance(family, n_candidates, seed=seed)
            # Simulate a semantic change to instance generation (e.g. a
            # different random draw) by perturbing the first candidate's
            # expected_utility -- changes the fingerprint (fingerprint()
            # hashes every candidate field) without touching this test's
            # own control flow.
            perturbed_first = dataclasses.replace(
                inst.candidates[0], expected_utility=inst.candidates[0].expected_utility + 1.0
            )
            new_candidates = (perturbed_first,) + inst.candidates[1:]
            return dataclasses.replace(inst, candidates=new_candidates)

        monkeypatch.setattr(fc_mod, "generate_instance", drifted_generate_instance)

        results = verify_frozen_controls(check_simulator_baseline=False)  # default: loads committed snapshot

        assert any(not r.all_match for r in results), (
            "a semantic change to generate_instance must be caught against the "
            "unchanged committed snapshot -- if this fails, the drift check is "
            "tautological again"
        )

    def test_FROZEN_CONTROLS_against_itself_still_available_as_an_explicit_opt_in(self):
        """The old self-consistency check is still reachable for a caller
        that explicitly wants it -- just no longer the silent default."""
        results = verify_frozen_controls(against=FROZEN_CONTROLS)
        assert all(r.all_match for r in results)


class TestSimulatorBaseline:
    def test_every_control_has_a_finite_simulator_baseline(self):
        import math

        for c in FROZEN_CONTROLS:
            assert not math.isnan(c.expected_simulator_expectation_qaoa_best_objective)
            assert not math.isnan(c.expected_simulator_cvar_qaoa_best_objective)

    def test_simulator_baseline_never_beats_the_exact_optimum(self):
        """A QAOA heuristic's best FEASIBLE objective can tie but never
        exceed Track A's exact optimum on the identical instance -- this
        is a basic sanity check on the pinned numbers themselves, not a
        re-derivation of EXP-B1's own analysis."""
        for c in FROZEN_CONTROLS:
            assert c.expected_simulator_expectation_qaoa_best_objective <= c.expected_exact_optimum + 1e-9
            assert c.expected_simulator_cvar_qaoa_best_objective <= c.expected_exact_optimum + 1e-9

    def test_verify_frozen_controls_checks_simulator_baseline_by_default(self):
        results = verify_frozen_controls()
        for r in results:
            assert r.simulator_baseline_matches is True
            assert r.all_match

    def test_verify_frozen_controls_can_skip_simulator_baseline_for_speed(self):
        results = verify_frozen_controls(check_simulator_baseline=False)
        for r in results:
            assert r.simulator_baseline_matches is True  # default True when skipped, not a false pass
            assert r.all_match

    def test_verify_frozen_controls_detects_a_wrong_simulator_baseline(self):
        import dataclasses

        tampered = dataclasses.replace(
            FROZEN_CONTROLS[0], expected_simulator_expectation_qaoa_best_objective=-999999.0
        )
        results = verify_frozen_controls((tampered,) + FROZEN_CONTROLS[1:])
        assert results[0].simulator_baseline_matches is False
        assert results[0].all_match is False
        for r in results[1:]:
            assert r.all_match
