"""Tests for `experiments.quantum1b.quantum_lab.charts`.

Covers the explicit regression requirements from the 2026-10-01
visualization/research-communication increment:
- every displayed chart corresponds to a real experiment/result
  (cross-checked against an independent direct read of the source JSON);
- no fabricated data series can enter the chart layer (AST-level guard:
  every chart builder's numeric series must come from a `_load(...)` call,
  never a hand-typed numeric literal list standing in for data);
- chart provenance remains traceable (`experiment_id` + `source_data` +
  `source_artifacts` on every chart);
- experiment evidence_level is carried from the registry, never
  hand-typed separately, so a chart cannot silently disagree with the
  scientific conclusion the registry states.
"""
from __future__ import annotations

import ast
import inspect
import json

import pytest

from experiments.quantum1b.quantum_lab.charts import (
    RESULTS_DIR,
    ChartSpec,
    ResearchResultComponent,
    build_all_charts,
    build_all_charts_tolerant,
    build_research_result_components,
    exp_a1_correctness_verification,
    exp_a2_classical_scaling,
    exp_b1_feasibility_comparison,
    exp_b2_shot_convergence,
    exp_b3_entanglement_growth,
    exp_c1_memory_capacity_per_delay,
    exp_c1_oos_mse_comparison,
    exp_d1_depth_inflation_with_replication,
    exp_e1_qmc_vs_mc_convergence,
    exp_e2_estimate_distribution,
    exp_e2_vs_e3_noise_sensitivity,
    exp_e4_depth_vs_k_scaling,
    exp_f1_cobyla_eval_count_by_strategy,
)
from experiments.quantum1b.quantum_lab.registry import QUANTUM_LAB_REGISTRY, get_experiment

_ALL_CHART_BUILDERS = (
    exp_a2_classical_scaling,
    exp_b1_feasibility_comparison,
    exp_b2_shot_convergence,
    exp_b3_entanglement_growth,
    exp_c1_oos_mse_comparison,
    exp_c1_memory_capacity_per_delay,
    exp_d1_depth_inflation_with_replication,
    exp_e1_qmc_vs_mc_convergence,
    exp_e2_estimate_distribution,
    exp_e2_vs_e3_noise_sensitivity,
    exp_e4_depth_vs_k_scaling,
    exp_f1_cobyla_eval_count_by_strategy,
)


class TestBuildAllCharts:
    def test_builds_all_twelve_charts_with_no_errors(self):
        charts, errors = build_all_charts_tolerant()
        assert errors == []
        assert len(charts) == 12

    def test_every_chart_is_a_chartspec_with_nonempty_series(self):
        for chart in build_all_charts():
            assert isinstance(chart, ChartSpec)
            assert chart.series
            for s in chart.series:
                assert len(s.x) == len(s.y)
                assert len(s.x) > 0
                if s.y_err is not None:
                    assert len(s.y_err) == len(s.y)

    def test_every_chart_cites_a_real_existing_source_artifact(self):
        from pathlib import Path

        repo_root = Path(__file__).parents[3]
        for chart in build_all_charts():
            assert chart.source_artifacts
            for rel_path in chart.source_artifacts:
                assert (repo_root / rel_path).is_file(), f"{chart.chart_id} cites missing {rel_path}"

    def test_to_dict_is_json_serializable(self):
        for chart in build_all_charts():
            json.dumps(chart.to_dict())

    def test_research_result_components_build_and_are_serializable(self):
        components = build_research_result_components()
        assert len(components) == 1
        for c in components:
            assert isinstance(c, ResearchResultComponent)
            json.dumps(c.to_dict())


class TestChartProvenance:
    """Regression guard: every chart must be traceable to experiment_id ->
    result artifact -> source data -> chart definition, per the explicit
    instruction that a chart which cannot be reproduced from the canonical
    artifact must never silently appear as if it were reproducible."""

    def test_every_chart_has_a_valid_experiment_id(self):
        registry_ids = {e.experiment_id for e in QUANTUM_LAB_REGISTRY}
        for chart in build_all_charts():
            assert chart.experiment_id in registry_ids, (
                f"{chart.chart_id} references unknown experiment_id {chart.experiment_id!r}"
            )

    def test_every_chart_has_nonempty_source_data_mapping(self):
        for chart in build_all_charts():
            assert chart.source_data, f"{chart.chart_id} has no source_data provenance mapping"
            for key, path_str in chart.source_data.items():
                assert isinstance(path_str, str) and path_str, f"{chart.chart_id}.source_data[{key!r}] is empty"

    def test_every_chart_has_a_specific_research_question(self):
        for chart in build_all_charts():
            assert chart.research_question
            assert chart.research_question.strip().endswith("?"), (
                f"{chart.chart_id}'s research_question should be phrased as a question"
            )

    def test_research_result_component_also_has_full_provenance(self):
        for c in build_research_result_components():
            assert c.experiment_id in {e.experiment_id for e in QUANTUM_LAB_REGISTRY}
            assert c.research_question.strip().endswith("?")
            assert c.source_artifacts


class TestEvidenceLevelMatchesRegistry:
    """Regression guard: a chart's evidence_level must never silently
    disagree with the registry's own stated evidence level for the same
    experiment -- both come from the exact same source
    (`get_experiment(id).evidence_level`), so this test also catches any
    future refactor that accidentally hand-types a different value."""

    def test_chart_evidence_level_matches_registry_for_every_chart(self):
        for chart in build_all_charts():
            assert chart.evidence_level == get_experiment(chart.experiment_id).evidence_level

    def test_research_result_component_evidence_level_matches_registry(self):
        for c in build_research_result_components():
            assert c.evidence_level == get_experiment(c.experiment_id).evidence_level


class TestNoFabricatedDataSeries:
    """AST-level guard: every chart-builder function's numeric series data
    must originate from `_load(...)` (reading a committed result JSON),
    never from a hand-typed literal list of numbers standing in for real
    data. Catches the exact failure mode this task was issued to prevent:
    a chart that LOOKS data-driven but is actually partly invented."""

    @pytest.mark.parametrize("builder", _ALL_CHART_BUILDERS, ids=lambda f: f.__name__)
    def test_builder_calls_load_and_has_no_large_hand_typed_numeric_literal_lists(self, builder):
        source = inspect.getsource(builder)
        tree = ast.parse(source)

        assert "_load(" in source, f"{builder.__name__} never calls _load() -- may not read real data"

        # Look for any list literal containing 4+ numeric constants -- the
        # shape a fabricated data series would take. Short literals (e.g.
        # [4, 6, 8] grid definitions, or [0, 1] axis ticks) are allowed --
        # they're parameter grids already used elsewhere (ks, shot_grid),
        # not invented RESULT values.
        suspicious = []
        for node in ast.walk(tree):
            if isinstance(node, ast.List):
                numeric_elts = [
                    e for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                ]
                if len(numeric_elts) >= 6:
                    suspicious.append([e.value for e in numeric_elts])
        assert not suspicious, (
            f"{builder.__name__} contains a suspiciously long hand-typed numeric literal list "
            f"(possible fabricated data series): {suspicious}"
        )


class TestIndividualChartsAgainstRawData:
    """Cross-checks specific values against an independent direct read of
    the same source JSON -- the real guard against a chart silently
    reading the wrong field or computing the wrong aggregate."""

    def test_exp_a1_component_matches_raw_data(self):
        with open(RESULTS_DIR / "track_a_results.json") as f:
            raw = json.load(f)["EXP-A1"]
        comp = exp_a1_correctness_verification()
        values = {m["label"]: m["value"] for m in comp.metrics}
        assert values["cells tested"] == raw["n_cells"]
        assert values["cells in agreement"] == raw["n_agree"]
        assert values["fraction agreeing"] == pytest.approx(raw["fraction_agree"])

    def test_exp_a2_classical_scaling_matches_raw_cells(self):
        with open(RESULTS_DIR / "track_a_results.json") as f:
            raw = json.load(f)
        cells_n8 = [c["runtime_s"] for c in raw["EXP-A2"]["cells"] if c["n"] == 8]
        expected_mean = sum(cells_n8) / len(cells_n8)

        chart = exp_a2_classical_scaling()
        mean_series = next(s for s in chart.series if "mean" in s.name)
        idx = mean_series.x.index(8)
        assert mean_series.y[idx] == pytest.approx(expected_mean)
        # Individual measurements series must contain every raw cell's runtime, unaggregated.
        raw_series = next(s for s in chart.series if "individual" in s.name)
        assert len(raw_series.x) == len(raw["EXP-A2"]["cells"])

    def test_exp_b1_feasibility_matches_raw_cell_summaries(self):
        with open(RESULTS_DIR / "track_b_exp_b1.json") as f:
            raw = json.load(f)["EXP-B1"]
        chart = exp_b1_feasibility_comparison()
        assert len(chart.series[0].x) == len(raw["cell_summaries"]) == 24
        target = next(c for c in raw["cell_summaries"] if c["family"] == "independent" and c["n"] == 4)
        label = "inde/n4"
        idx = chart.series[0].x.index(label)
        assert chart.series[0].y[idx] == pytest.approx(target["expectation_qaoa_median_feasible_probability"])
        assert chart.series[1].y[idx] == pytest.approx(target["cvar_qaoa_best_alpha_median_feasible_probability"])

    def test_exp_b2_shot_convergence_matches_raw_shots_summary(self):
        with open(RESULTS_DIR / "track_b_exp_b2.json") as f:
            raw = json.load(f)["EXP-B2"]
        chart = exp_b2_shot_convergence()
        mean_series = chart.series[0]
        for row in raw["shots_summary"]:
            idx = mean_series.x.index(row["shots"])
            assert mean_series.y[idx] == pytest.approx(row["mean_abs_error"])
            assert mean_series.y_err[idx] == pytest.approx(row["std_abs_error"])

    def test_exp_b3_entanglement_matches_a_specific_cell(self):
        with open(RESULTS_DIR / "track_b_exp_b3.json") as f:
            raw = json.load(f)
        target = next(c for c in raw["EXP-B3"]["cells"] if c["p"] == 1 and c["n_qubits"] == 9)
        chart = exp_b3_entanglement_growth()
        p1_series = next(s for s in chart.series if s.name == "p=1")
        idx = p1_series.x.index(9)
        assert p1_series.y[idx] == pytest.approx(target["entropy_fraction_of_max"])

    def test_exp_c1_oos_mse_matches_raw_metrics_and_baselines(self):
        with open(RESULTS_DIR / "track_c_results.json") as f:
            raw = json.load(f)
        chart = exp_c1_oos_mse_comparison()
        esn_series = next(s for s in chart.series if "ESN" in s.name)
        idx = esn_series.x.index(6)
        assert esn_series.y[idx] == pytest.approx(raw["metrics"]["esn_mean_oos_mse_D6"])
        persistence_ref = next(r for r in chart.reference_lines if "persistence" in r.label)
        assert persistence_ref.value == pytest.approx(raw["metrics"]["persistence_mean_oos_mse"])

    def test_exp_c1_memory_capacity_per_delay_matches_raw_data(self):
        with open(RESULTS_DIR / "track_c_results.json") as f:
            raw = json.load(f)
        esn_raw = raw["comparison"]["esn"]["4"][0]["extra"]["memory_capacity"]["per_delay"]
        chart = exp_c1_memory_capacity_per_delay()
        esn_series = next(s for s in chart.series if "ESN" in s.name)
        assert esn_series.x == [r["delay"] for r in esn_raw]
        assert esn_series.y == pytest.approx([r["capacity"] for r in esn_raw])

    def test_exp_d1_depth_inflation_matches_the_seed_replication_json(self):
        with open(RESULTS_DIR / "track_d_exp_d1_seed_replication.json") as f:
            raw = json.load(f)
        chart = exp_d1_depth_inflation_with_replication()
        raw_section = raw["EXP-D1-seed-replication"]
        summary_series = next(s for s in chart.series if "mean" in s.name)
        for i, n_candidates in enumerate(sorted(int(k) for k in raw_section["by_rung"])):
            rung = raw_section["by_rung"][str(n_candidates)]
            assert summary_series.y[i] == pytest.approx(rung["mean_ratio"])
            assert summary_series.y_err[i] == pytest.approx(rung["std_ratio"])
        raw_series = next(s for s in chart.series if "individual" in s.name)
        assert len(raw_series.x) == len(raw_section["rows"])

    def test_exp_e1_convergence_matches_raw_audit(self):
        with open(RESULTS_DIR / "track_e_exp_e1.json") as f:
            raw = json.load(f)["EXP-E1"]["default_book"]["raw_audit"]["A_qmc_vs_mc"]
        chart = exp_e1_qmc_vs_mc_convergence()
        assert chart.series[0].x == raw["Ns"]
        assert chart.series[0].y == raw["mc_rmse"]
        assert chart.series[1].y == raw["qmc_rmse"]

    def test_exp_e2_distribution_matches_raw_trials(self):
        with open(RESULTS_DIR / "track_e_exp_e2.json") as f:
            raw = json.load(f)["EXP-E2"]
        chart = exp_e2_estimate_distribution()
        iqae_series = next(s for s in chart.series if "IQAE" in s.name)
        assert iqae_series.y == [t["a_hat"] for t in raw["iqae"]["trials"]]
        true_value_ref = chart.reference_lines[0]
        assert true_value_ref.value == pytest.approx(raw["true_value_numeric"])

    def test_exp_e2_vs_e3_matches_both_raw_files(self):
        with open(RESULTS_DIR / "track_e_exp_e2.json") as f:
            e2 = json.load(f)["EXP-E2"]
        with open(RESULTS_DIR / "track_e_exp_e3.json") as f:
            e3 = json.load(f)["EXP-E3"]
        chart = exp_e2_vs_e3_noise_sensitivity()
        assert chart.series[0].y[0] == pytest.approx(e2["iqae"]["empirical_ci_coverage"])
        assert chart.series[0].y[1] == pytest.approx(e3["iqae"]["empirical_ci_coverage"])

    def test_exp_e4_depth_vs_k_matches_the_opt_level_comparison_json(self):
        with open(RESULTS_DIR / "track_e_exp_e4_opt_level_comparison.json") as f:
            raw = json.load(f)["EXP-E4-opt-level-comparison"]
        chart = exp_e4_depth_vs_k_scaling()
        level1_series = next(s for s in chart.series if s.name == "optimization_level=1")
        k128_idx = level1_series.x.index(128)
        assert level1_series.y[k128_idx] == raw["by_optimization_level"]["1"]["circuits"]["Q128A"]["depth"]

    def test_exp_f1_eval_count_matches_raw_stats_and_shows_real_spread(self):
        with open(RESULTS_DIR / "track_f_exp_f1.json") as f:
            raw = json.load(f)["EXP-F1"]["instances"]["independent_n4"]
        chart = exp_f1_cobyla_eval_count_by_strategy()
        raw_mean = raw["cold_start_random_p1"]["stats"]["mean_n_evaluations"]
        idx = chart.series[0].x.index("cold_start (p=1)")
        assert chart.series[0].y[idx] == pytest.approx(raw_mean)
        # Error bars must be non-trivial (real spread), not a hardcoded 0.
        assert chart.series[0].y_err[idx] > 0
