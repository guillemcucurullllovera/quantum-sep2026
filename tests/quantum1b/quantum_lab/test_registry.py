"""Tests for `experiments.quantum1b.quantum_lab.registry`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.quantum1b.quantum_lab.readiness_state import ReadinessState
from experiments.quantum1b.quantum_lab.registry import (
    QUANTUM_LAB_REGISTRY,
    by_track,
    get_experiment,
)

REPO_ROOT = Path(__file__).parents[3]
REQUIRED_TEXT_FIELDS = (
    "problem", "hypothesis", "classical_baseline", "quantum_formulation",
    "simulator_noise_config", "result", "evidence_level", "conclusion",
    "limitations", "next_experiment", "source_doc",
)


class TestRegistryShape:
    def test_registry_is_nonempty(self):
        assert len(QUANTUM_LAB_REGISTRY) > 0

    def test_every_experiment_id_is_unique(self):
        ids = [e.experiment_id for e in QUANTUM_LAB_REGISTRY]
        assert len(ids) == len(set(ids)), f"duplicate experiment_id(s): {ids}"

    @pytest.mark.parametrize("exp", QUANTUM_LAB_REGISTRY, ids=lambda e: e.experiment_id)
    def test_every_required_text_field_is_nonempty(self, exp):
        for field_name in REQUIRED_TEXT_FIELDS:
            value = getattr(exp, field_name)
            assert isinstance(value, str) and value.strip(), (
                f"{exp.experiment_id}.{field_name} must be a non-empty string"
            )

    @pytest.mark.parametrize("exp", QUANTUM_LAB_REGISTRY, ids=lambda e: e.experiment_id)
    def test_at_least_one_result_artifact_path(self, exp):
        assert len(exp.result_artifact_paths) >= 1

    @pytest.mark.parametrize("exp", QUANTUM_LAB_REGISTRY, ids=lambda e: e.experiment_id)
    def test_result_artifact_paths_exist_on_disk(self, exp):
        """Every path cited in the registry must point at a real,
        already-committed result file -- the whole point of this registry
        is to wrap existing evidence, never to assert an experiment ran
        without a resolvable artifact backing that claim."""
        for rel_path in exp.result_artifact_paths:
            full = REPO_ROOT / rel_path
            assert full.is_file(), f"{exp.experiment_id} cites missing artifact {rel_path}"

    @pytest.mark.parametrize("exp", QUANTUM_LAB_REGISTRY, ids=lambda e: e.experiment_id)
    def test_result_artifact_paths_are_valid_json(self, exp):
        for rel_path in exp.result_artifact_paths:
            full = REPO_ROOT / rel_path
            with open(full) as f:
                json.load(f)  # raises if malformed

    @pytest.mark.parametrize("exp", QUANTUM_LAB_REGISTRY, ids=lambda e: e.experiment_id)
    def test_source_doc_exists_on_disk(self, exp):
        full = REPO_ROOT / exp.source_doc
        assert full.is_file(), f"{exp.experiment_id} cites missing source_doc {exp.source_doc}"

    @pytest.mark.parametrize("exp", QUANTUM_LAB_REGISTRY, ids=lambda e: e.experiment_id)
    def test_no_quantum_advantage_or_financial_edge_language_leaks_into_conclusions(self, exp):
        """A blunt guard, not a substitute for careful writing: these
        specific phrases must never appear asserting a positive claim in a
        conclusion field (they are fine appearing as part of an explicit
        negation, e.g. 'no quantum-advantage claim')."""
        lowered = exp.conclusion.lower()
        for forbidden in ("proves quantum advantage", "confirms financial edge", "beats classical hardware"):
            assert forbidden not in lowered


class TestScientificConclusionsUnchangedByVisualizationWork:
    """Regression guard for the 2026-10-01 visualization/research-
    communication increment, which explicitly must NOT alter experimental
    conclusions, evidence levels, or readiness states -- only charts.py
    and command_center_export.py were touched for that work; this test
    proves the registry's own scientific content (which only that work's
    CHARTS read from, never write to) is byte-for-byte unchanged, via a
    hash of every experiment's conclusion+evidence_level+readiness_state."""

    def test_conclusion_evidence_and_readiness_hash_unchanged(self):
        import hashlib

        payload = "|".join(
            f"{e.experiment_id}:{e.conclusion}:{e.evidence_level}:{e.readiness_state().value}"
            for e in QUANTUM_LAB_REGISTRY
        )
        actual_hash = hashlib.sha256(payload.encode()).hexdigest()
        expected_hash = "3d7da3044513c2be1712745cee913cb9c2b79e55fae038ff20f8e450d9871f2f"
        assert actual_hash == expected_hash, (
            "A registry experiment's conclusion, evidence_level, or readiness_state changed -- "
            "the visualization increment must never alter scientific content. If this change is "
            "deliberate and reviewed, update expected_hash to match, with a clear commit message "
            "explaining what changed and why."
        )


class TestLookupHelpers:
    def test_get_experiment_returns_the_right_record(self):
        exp = get_experiment("EXP-B3")
        assert exp.track == "B"
        assert "CSIM" in exp.conclusion

    def test_get_experiment_unknown_id_raises_key_error(self):
        with pytest.raises(KeyError):
            get_experiment("EXP-ZZZ-DOES-NOT-EXIST")

    def test_by_track_filters_correctly(self):
        track_b = by_track("B")
        assert len(track_b) >= 3  # B1, B2, B3 at minimum
        assert all(e.track == "B" for e in track_b)

    def test_by_track_unknown_track_returns_empty(self):
        assert by_track("Z") == ()


class TestReadinessStateIntegration:
    @pytest.mark.parametrize("exp", QUANTUM_LAB_REGISTRY, ids=lambda e: e.experiment_id)
    def test_every_experiment_resolves_to_a_readiness_state_without_error(self, exp):
        state = exp.readiness_state()
        assert isinstance(state, ReadinessState)

    @pytest.mark.parametrize("exp", QUANTUM_LAB_REGISTRY, ids=lambda e: e.experiment_id)
    def test_no_experiment_is_currently_hardware_ready_or_executed(self, exp):
        """The load-bearing safety assertion of this entire module: given
        the real current repo state (no credentials, capstone NOT_READY, no
        review sign-off), NOT ONE experiment may report HARDWARE_READY or
        HARDWARE_EXECUTED. If this ever fails, something in the gating
        logic or the lab-wide facts has gone wrong -- it must be fixed
        before trusting this registry's readiness output again."""
        state = exp.readiness_state()
        assert state not in (ReadinessState.HARDWARE_READY, ReadinessState.HARDWARE_EXECUTED)

    def test_exp_d1_reaches_hardware_ready_for_review(self):
        """EXP-D1 is the one experiment in this registry with both a
        noise-validated result AND the hardware_readiness_pathway_exists
        flag set -- it should be the strongest state currently reachable,
        given frozen_controls_verified is True in a clean checkout."""
        exp = get_experiment("EXP-D1")
        assert exp.readiness_state() == ReadinessState.HARDWARE_READY_FOR_REVIEW

    def test_exp_c1_caps_at_noise_validated_lacking_a_hardware_pathway(self):
        exp = get_experiment("EXP-C1")
        assert exp.hardware_readiness_pathway_exists is False
        assert exp.readiness_state() in (
            ReadinessState.RESEARCH,
            ReadinessState.SIMULATOR_VALIDATED,
            ReadinessState.NOISE_VALIDATED,
        )
