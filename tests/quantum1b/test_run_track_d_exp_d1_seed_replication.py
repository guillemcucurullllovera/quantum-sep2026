"""Tests for `experiments.quantum1b.run_track_d_exp_d1_seed_replication`.

Deliberately does NOT re-run the actual transpile sweep in this fast suite
(the script itself is the on-demand job, ~16s measured -- same
smoke-test-vs-full-run convention used throughout this package). Sanity-
checks the already-committed result JSON, if present, without re-running
it as a side effect of the test suite."""
from __future__ import annotations

import json

import pytest

import experiments.quantum1b.run_track_d_exp_d1_seed_replication as mod


class TestModuleStructure:
    def test_has_main_and_expected_constants(self):
        assert callable(mod.main)
        assert mod.N_CANDIDATES_RUNGS == (3, 6)
        assert len(mod.SEEDS) >= 3  # the review's own "3+ seeds per rung" minimum
        assert mod.OPTIMIZATION_LEVEL == 1  # the corrected, P1-2 convention


class TestCommittedResultIfPresent:
    def test_result_shape_and_internal_consistency(self):
        if not mod.RESULTS_PATH.is_file():
            pytest.skip("track_d_exp_d1_seed_replication.json has not been generated in this checkout")

        with open(mod.RESULTS_PATH) as f:
            data = json.load(f)

        section = data["EXP-D1-seed-replication"]
        assert len(section["rows"]) == len(mod.N_CANDIDATES_RUNGS) * len(section["seeds"])

        for n_candidates in mod.N_CANDIDATES_RUNGS:
            rung = section["by_rung"][str(n_candidates)]
            assert rung["n_seeds"] == len(section["seeds"])
            assert rung["min_ratio"] <= rung["mean_ratio"] <= rung["max_ratio"]
            assert rung["std_ratio"] >= 0.0

        # The larger rung (n_candidates=6, more qubits) must have a higher
        # mean depth-inflation ratio than the smaller rung (n_candidates=3)
        # -- this is the actual directional claim being tested.
        mean_n3 = section["by_rung"][str(mod.N_CANDIDATES_RUNGS[0])]["mean_ratio"]
        mean_n6 = section["by_rung"][str(mod.N_CANDIDATES_RUNGS[1])]["mean_ratio"]
        assert mean_n6 > mean_n3

    def test_seed_zero_matches_the_original_single_seed_ladder_measurement(self):
        """The strongest possible consistency check: this script's own
        seed=0 row, for each rung, must reproduce the EXACT depth/ratio
        values the original (pre-replication) EXP-D1 ladder reported --
        confirming this is a genuine extension of the same measurement,
        not a different one."""
        if not mod.RESULTS_PATH.is_file():
            pytest.skip("track_d_exp_d1_seed_replication.json has not been generated in this checkout")

        with open(mod.RESULTS_PATH) as f:
            data = json.load(f)

        rows = data["EXP-D1-seed-replication"]["rows"]
        seed0_n3 = next(r for r in rows if r["n_candidates"] == 3 and r["seed"] == 0)
        seed0_n6 = next(r for r in rows if r["n_candidates"] == 6 and r["seed"] == 0)

        assert seed0_n3["fake_kingston_depth"] == 176
        assert seed0_n3["ideal_depth"] == 14
        assert seed0_n6["fake_kingston_depth"] == 408
        assert seed0_n6["ideal_depth"] == 20
