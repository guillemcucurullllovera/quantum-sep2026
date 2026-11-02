"""Tests for `experiments.quantum1b.calibrate_transpile_optimization_level`
(P1-2 remediation). Deliberately does NOT re-run the actual FakeKingston
calibration in this fast suite (the script itself is the on-demand job,
~15s, same smoke-test-vs-full-run convention used throughout this
package) -- this test only checks the module is well-formed and does not
reintroduce the duplicate-expensive-build bug caught and fixed while
writing it (calling `fake_kingston_backend()` twice)."""
from __future__ import annotations

import ast


class TestCalibrationScriptStructure:
    def test_does_not_call_fake_kingston_backend_more_than_once(self):
        """Regression guard for a real bug caught while writing this
        script: an earlier version called fake_kingston_backend() a
        second time just to read its .name, duplicating an ~11s build and
        causing a MemoryError under a bounded ulimit. The fix reuses the
        already-built `backend` variable."""
        import experiments.quantum1b.calibrate_transpile_optimization_level as mod

        tree = ast.parse(open(mod.__file__).read())
        call_count = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "fake_kingston_backend"
        )
        assert call_count == 1, f"fake_kingston_backend() called {call_count} times, expected exactly 1"

    def test_module_has_main_and_expected_constants(self):
        import experiments.quantum1b.calibrate_transpile_optimization_level as mod

        assert callable(mod.main)
        assert mod.OPT_LEVELS == (1, 0)
        assert mod.N_CANDIDATES == 8
