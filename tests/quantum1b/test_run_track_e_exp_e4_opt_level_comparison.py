"""Tests for `experiments.quantum1b.run_track_e_exp_e4_opt_level_comparison`.

Deliberately does NOT re-run the actual FakeKingston transpile sweep in
this fast suite (the script itself is the on-demand job, ~11s measured --
same smoke-test-vs-full-run convention used throughout this package).
Structure/fit-math checks only, plus a sanity check against the already-
committed result JSON (if present) rather than re-running."""
from __future__ import annotations

import ast
import json
from pathlib import Path


class TestReusesSharedDecomposeHelper:
    def test_imports_decompose_does_not_redefine_it(self):
        """Confirms this script reuses `algorithms.qae_iqae.iqae._decompose`
        (the shared circuit-decomposition helper) rather than reimplementing
        it, even though it reimplements the small transpile-and-measure loop
        itself (see the module docstring for why -- avoiding an edit to
        shared production code outside this session's established scope)."""
        import experiments.quantum1b.run_track_e_exp_e4_opt_level_comparison as mod

        tree = ast.parse(open(mod.__file__).read())
        imported_names = set()
        defined_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_names.update(a.name for a in node.names)
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)

        assert "_decompose" in imported_names
        assert "_decompose" not in defined_names
        # _grover_metadata itself IS reimplemented here (as
        # _grover_metadata_at_opt_level) -- deliberately, per the module
        # docstring -- so this guard only checks _decompose, not both.
        assert "_grover_metadata_at_opt_level" in defined_names


class TestLinearFitMath:
    def test_perfect_linear_data_recovers_exact_slope_intercept(self):
        import experiments.quantum1b.run_track_e_exp_e4_opt_level_comparison as mod

        ks_list = [0, 1, 2, 4, 8, 16, 32]
        true_slope, true_intercept = 850.0, 355.0
        depths = [true_intercept + true_slope * k for k in ks_list]

        slope, intercept = mod._linear_fit(ks_list, depths)

        assert abs(slope - true_slope) < 1e-6
        assert abs(intercept - true_intercept) < 1e-6


class TestCommittedResultIfPresent:
    """Sanity-checks the already-committed result JSON, if one exists in
    this checkout, WITHOUT re-running the (fast but non-trivial) FakeKingston
    sweep as a side effect of the test suite -- same convention as
    `test_calibrate_transpile_optimization_level.py`'s own structure-only
    approach. Skipped cleanly if the file has never been generated."""

    def test_opt_level_1_is_not_deeper_than_opt_level_0_at_every_k(self):
        import pytest

        import experiments.quantum1b.run_track_e_exp_e4_opt_level_comparison as mod

        if not mod.RESULTS_PATH.is_file():
            pytest.skip("track_e_exp_e4_opt_level_comparison.json has not been generated in this checkout")

        with open(mod.RESULTS_PATH) as f:
            data = json.load(f)

        section = data["EXP-E4-opt-level-comparison"]
        circuits_0 = section["by_optimization_level"]["0"]["circuits"]
        circuits_1 = section["by_optimization_level"]["1"]["circuits"]
        for k in section["ks"]:
            key = f"Q{k}A"
            assert circuits_1[key]["depth"] <= circuits_0[key]["depth"], (
                f"opt_level=1 depth at k={k} ({circuits_1[key]['depth']}) exceeded "
                f"opt_level=0 ({circuits_0[key]['depth']}) -- the measured trend broke"
            )

    def test_comparison_conclusion_matches_the_measured_depths(self):
        import pytest

        import experiments.quantum1b.run_track_e_exp_e4_opt_level_comparison as mod

        if not mod.RESULTS_PATH.is_file():
            pytest.skip("track_e_exp_e4_opt_level_comparison.json has not been generated in this checkout")

        with open(mod.RESULTS_PATH) as f:
            data = json.load(f)

        section = data["EXP-E4-opt-level-comparison"]
        comparison = section["comparison_at_largest_k"]
        assert comparison["opt_level_1_is_shallower_or_equal"] == (
            comparison["depth_opt_level_1"] <= comparison["depth_opt_level_0"]
        )
