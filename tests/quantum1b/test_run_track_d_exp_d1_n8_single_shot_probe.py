"""Tests for `experiments.quantum1b.run_track_d_exp_d1_n8_single_shot_probe`.

Deliberately does NOT run `main()` in this fast suite -- on this host it
does not complete within any bounded attempt tried so far (see the
module's own docstring and `results/track_d_exp_d1_n8_single_shot_probe.json`'s
own `manifest_provenance` field for the full, honest account). Structure-
only checks, plus a sanity check against the already-committed result."""
from __future__ import annotations

import json

import experiments.quantum1b.run_track_d_exp_d1_n8_single_shot_probe as mod


class TestModuleStructure:
    def test_has_main_and_expected_constants(self):
        assert callable(mod.main)
        assert mod.N_CANDIDATES == 8
        assert mod.SHOTS == 1024
        assert mod.OPTIMIZATION_LEVEL == 1


class TestCommittedResultIsHonestAboutNonCompletion:
    def test_result_discloses_non_completion_rather_than_fabricating_a_number(self):
        if not mod.RESULTS_PATH.is_file():
            import pytest

            pytest.skip("track_d_exp_d1_n8_single_shot_probe.json has not been generated in this checkout")

        with open(mod.RESULTS_PATH) as f:
            data = json.load(f)

        section = data["EXP-D1-n8-single-shot-probe"]
        # The actual single-execution time must NOT be fabricated as a
        # point estimate -- it must be explicitly None/absent, with a
        # disclosed lower bound instead.
        assert section["single_execution_s"] is None
        assert section["single_execution_lower_bound_s"] > 0
        assert "manifest_provenance" in data

    def test_backend_build_and_transpile_were_genuinely_measured(self):
        if not mod.RESULTS_PATH.is_file():
            import pytest

            pytest.skip("track_d_exp_d1_n8_single_shot_probe.json has not been generated in this checkout")

        with open(mod.RESULTS_PATH) as f:
            data = json.load(f)

        section = data["EXP-D1-n8-single-shot-probe"]
        assert section["backend_build_s"] > 0
        assert section["transpile_s"] > 0
        assert section["transpiled_depth"] > 0
