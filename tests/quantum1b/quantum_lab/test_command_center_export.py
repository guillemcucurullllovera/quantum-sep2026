"""Tests for `experiments.quantum1b.quantum_lab.command_center_export`.

No real Command Center repo is touched anywhere -- this module only writes
under `experiments/quantum1b/results/` in this worktree, which these tests
verify structurally (path assertions), not just by trusting the docstring.
"""
from __future__ import annotations

import json

from experiments.quantum1b.quantum_lab.command_center_export import (
    OUTPUT_PATH,
    build_command_center_export,
    main,
)
from experiments.quantum1b.quantum_lab.readiness_state import ReadinessState
from experiments.quantum1b.quantum_lab.registry import QUANTUM_LAB_REGISTRY


class TestBuildCommandCenterExport:
    def test_returns_a_json_serializable_dict(self):
        export = build_command_center_export()
        json.dumps(export)  # raises on anything non-serializable

    def test_no_quantum_advantage_or_financial_edge_claim_flags(self):
        export = build_command_center_export()
        assert export["no_quantum_advantage_claim"] is True
        assert export["no_financial_edge_claim"] is True

    def test_includes_every_registry_experiment(self):
        export = build_command_center_export()
        exported_ids = {e["experiment_id"] for e in export["experiments"]}
        registry_ids = {e.experiment_id for e in QUANTUM_LAB_REGISTRY}
        assert exported_ids == registry_ids

    def test_completed_experiments_count_matches_registry_size(self):
        export = build_command_center_export()
        assert export["completed_experiments_count"] == len(QUANTUM_LAB_REGISTRY)

    def test_no_experiment_reports_an_active_status(self):
        """Every registry entry has already produced a result -- none
        should be labeled anything other than COMPLETED, and
        active_experiments must be empty, since this export does not
        fabricate in-progress work that doesn't exist."""
        export = build_command_center_export()
        assert export["active_experiments"] == []
        assert all(e["status"] == "COMPLETED" for e in export["experiments"])

    def test_no_exported_experiment_is_hardware_ready_or_executed(self):
        export = build_command_center_export()
        forbidden = {ReadinessState.HARDWARE_READY.value, ReadinessState.HARDWARE_EXECUTED.value}
        for exp in export["experiments"]:
            assert exp["hardware_readiness_state"] not in forbidden

    def test_every_exported_experiment_has_all_required_display_fields(self):
        required_fields = {
            "experiment_id", "track", "status", "problem", "hypothesis",
            "classical_baseline", "quantum_formulation", "simulator_noise_config",
            "result", "evidence_level", "conclusion", "limitations", "next_experiment",
            "hardware_readiness_state", "has_ideal_simulator_result",
            "has_noise_validated_result", "hardware_readiness_pathway_exists",
            "result_artifact_paths", "source_doc",
        }
        export = build_command_center_export()
        for exp in export["experiments"]:
            assert required_fields.issubset(exp.keys())

    def test_simulator_and_noise_and_hardware_readiness_are_separately_structured(self):
        """Command Center integration audit (2026-10-01): simulator/noise/
        hardware-readiness must each be independently consumable, not only
        conflated inside a free-text `simulator_noise_config` string."""
        export = build_command_center_export()
        exp_d1 = next(e for e in export["experiments"] if e["experiment_id"] == "EXP-D1")
        assert exp_d1["has_ideal_simulator_result"] is True
        assert exp_d1["has_noise_validated_result"] is True
        assert exp_d1["hardware_readiness_pathway_exists"] is True
        assert isinstance(exp_d1["hardware_readiness_state"], str)

        exp_c1 = next(e for e in export["experiments"] if e["experiment_id"] == "EXP-C1")
        assert exp_c1["hardware_readiness_pathway_exists"] is False

    def test_capstone_decision_present_and_matches_the_constant(self):
        from experiments.quantum1b.quantum_lab.readiness_state import CAPSTONE_DECISION

        export = build_command_center_export()
        assert export["capstone_decision"] == CAPSTONE_DECISION
        assert export["capstone_decision"] == "NOT_READY"

    def test_remaining_blockers_present_and_structured(self):
        export = build_command_center_export()
        assert len(export["remaining_blockers"]) >= 5
        for blocker in export["remaining_blockers"]:
            assert blocker["blocker"]
            assert blocker["category"]
            assert blocker["resolvable_by"]

    def test_provenance_section_present(self):
        export = build_command_center_export()
        prov = export["provenance"]
        assert prov["generator_module"] == "experiments.quantum1b.quantum_lab.command_center_export"
        # git_head_at_generation may be None if git is unavailable, but the key must exist.
        assert "git_head_at_generation" in prov

    def test_reproducibility_section_present(self):
        export = build_command_center_export()
        repro = export["reproducibility"]
        assert "pytest" in repro["test_command"]
        assert repro["last_verified_test_result"]
        assert repro["last_verified_approximately_at_git_head"]

    def test_lab_wide_gates_are_present_and_boolean(self):
        export = build_command_center_export()
        gates = export["lab_wide_hardware_readiness_gates"]
        assert gates and all(isinstance(v, bool) for v in gates.values())

    def test_charts_section_present_and_nonempty(self):
        """Redesigned 2026-10-01 (visualization increment): 12 charts, up
        from the original 8, covering 11 of the 12 registered experiments
        with a real chart each (EXP-A1 is the one exception -- a binary
        correctness verification with no natural continuous axis, shown
        as a research_result_component instead, not forced into a chart)."""
        export = build_command_center_export()
        assert len(export["charts"]) == 12
        assert export["chart_generation_errors"] == []
        for chart in export["charts"]:
            assert "chart_id" in chart
            assert "experiment_id" in chart
            assert "research_question" in chart
            assert "series" in chart
            assert "source_data" in chart
            assert "evidence_level" in chart

    def test_research_result_components_section_present(self):
        export = build_command_center_export()
        assert len(export["research_result_components"]) == 1
        comp = export["research_result_components"][0]
        assert comp["experiment_id"] == "EXP-A1"
        assert "metrics" in comp

    def test_every_registered_experiment_has_a_chart_or_a_result_component(self):
        """No experiment should be silently unrepresented in the
        visualization layer -- each of the 12 must appear in either
        `charts` or `research_result_components`."""
        from experiments.quantum1b.quantum_lab.registry import QUANTUM_LAB_REGISTRY

        export = build_command_center_export()
        covered = {c["experiment_id"] for c in export["charts"]}
        covered |= {c["experiment_id"] for c in export["research_result_components"]}
        registered = {e.experiment_id for e in QUANTUM_LAB_REGISTRY}
        assert covered == registered, f"uncovered experiments: {registered - covered}"

    def test_flagship_prototype_section_present(self):
        export = build_command_center_export()
        flagship = export["flagship_end_to_end_prototype"]
        assert flagship["module"] == "experiments.quantum1b.quantum_lab.candidate_trade_prototype"
        # latest_result may be None if the prototype has never been run in
        # this checkout, or a dict if it has -- both are valid, this just
        # checks the key exists and the type is one of the two.
        assert flagship["latest_result"] is None or isinstance(flagship["latest_result"], dict)


class TestMainWritesOutput:
    def test_main_writes_to_quantum1b_results_never_elsewhere(self):
        main()
        assert OUTPUT_PATH.is_file()
        assert OUTPUT_PATH.parent.name == "results"
        assert OUTPUT_PATH.parent.parent.name == "quantum1b"
        assert "heimdall-command-center" not in str(OUTPUT_PATH)
        assert "heimdall-v2" not in str(OUTPUT_PATH)
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        assert data["lab_name"] == "Heimdall Quantum Lab"
