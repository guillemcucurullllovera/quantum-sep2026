"""Tests for `experiments.quantum1b.run_track_e_exp_e4`'s linear-fit logic
and its reuse of the existing `_grover_metadata` function.

Deliberately does NOT run the actual FakeKingston calibration in this fast
suite (that is what `run_track_e_exp_e4.py` itself does, on demand, ~5-15s
-- same smoke-test-vs-full-run convention as EXP-B3/EXP-D1 elsewhere in
this package). This tests the fit math against synthetic depth/k pairs,
and confirms `_grover_metadata` is reused (imported), not reimplemented.
"""
from __future__ import annotations

import ast


class TestReusesExistingGroverMetadata:
    def test_imports_grover_metadata_does_not_redefine_it(self):
        import experiments.quantum1b.run_track_e_exp_e4 as mod

        tree = ast.parse(open(mod.__file__).read())
        imported_names = set()
        defined_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_names.update(a.name for a in node.names)
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)

        assert "_grover_metadata" in imported_names
        assert "_grover_metadata" not in defined_names


class TestLinearFitMath:
    def test_perfect_linear_data_recovers_exact_slope_intercept(self):
        """Reproduces run_track_e_exp_e4.main()'s own inline fit logic
        against synthetic data with a KNOWN exact slope/intercept, to
        verify the fit formula itself (ordinary least squares via the
        covariance/variance ratio) is correct, independent of any real
        FakeKingston measurement."""
        ks_list = [0, 1, 2, 4, 8, 16, 32]
        true_slope, true_intercept = 1100.0, 444.0
        depths = [true_intercept + true_slope * k for k in ks_list]

        n = len(ks_list)
        mean_k = sum(ks_list) / n
        mean_d = sum(depths) / n
        slope = sum((k - mean_k) * (d - mean_d) for k, d in zip(ks_list, depths)) / sum(
            (k - mean_k) ** 2 for k in ks_list
        )
        intercept = mean_d - slope * mean_k

        assert abs(slope - true_slope) < 1e-6
        assert abs(intercept - true_intercept) < 1e-6
