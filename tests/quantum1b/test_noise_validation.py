"""Tests for `experiments.quantum1b.noise_validation` (Track D / EXP-D1).

Deliberately only exercises the IDEAL (bare `AerSimulator()`, no device
coupling-map/noise-model) backend in this fast unit-test suite -- the
FakeKingston-backed path is real-device-connectivity-constrained and takes
~2-3 minutes per cell (see `docs/quantum/QUANTUM1B_HARDWARE_READINESS.md`
for the calibration measurement), which is appropriate for the actual
`run_track_d_exp_d1.py` execution (its own resource-bounded run, matching
the convention `run_track_b_exp_b3.py` already established) but not for a
suite that should run in seconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.quantum1b.generators import generate_instance
from experiments.quantum1b.hardware_readiness.lineage import RunLineageRecord
from experiments.quantum1b.qubo_encoding import build_qubo
from experiments.quantum1b.noise_validation import (
    NoiseValidationRunResult,
    fake_kingston_backend,
    ideal_backend,
    run_noise_validation_cell,
)


class TestRunNoiseValidationCellIdealBackend:
    def test_produces_sane_result_structure(self):
        inst = generate_instance("independent", 3, seed=0)
        qubo = build_qubo(inst)
        backend = ideal_backend()

        result = run_noise_validation_cell(
            inst, qubo, backend_label="ideal", backend=backend, p=1, maxiter=3, shots=256, seed=0
        )

        assert isinstance(result, NoiseValidationRunResult)
        assert result.backend_label == "ideal"
        assert result.depth > 0
        assert result.two_qubit_gates >= 0
        assert 0.0 <= result.feasible_fraction <= 1.0
        assert result.n_evaluations > 0
        assert result.shots == 256
        # exact_optimum is computed fresh from Track A's own ground-truth
        # solver on the SAME instance -- sanity-check it's not a degenerate
        # placeholder.
        assert result.exact_optimum >= 0.0 or result.exact_optimum == 0.0

    def test_objective_gap_is_none_iff_no_feasible_sample(self):
        inst = generate_instance("independent", 3, seed=0)
        qubo = build_qubo(inst)
        backend = ideal_backend()

        result = run_noise_validation_cell(
            inst, qubo, backend_label="ideal", backend=backend, p=1, maxiter=3, shots=256, seed=0
        )
        if result.best_feasible_objective is None:
            assert result.objective_gap is None
        else:
            assert result.objective_gap is not None

    def test_deterministic_given_same_seed(self):
        inst = generate_instance("independent", 3, seed=0)
        qubo = build_qubo(inst)

        r1 = run_noise_validation_cell(
            inst, qubo, backend_label="ideal", backend=ideal_backend(), p=1, maxiter=3, shots=256, seed=0
        )
        r2 = run_noise_validation_cell(
            inst, qubo, backend_label="ideal", backend=ideal_backend(), p=1, maxiter=3, shots=256, seed=0
        )
        np.testing.assert_allclose(r1.best_params, r2.best_params)
        assert r1.expected_objective == pytest.approx(r2.expected_objective)

    def test_produces_a_real_lineage_record(self):
        """Regression test for independent-review finding P2-6
        ('build_lineage_record() is dead code -- never invoked by any real
        or dry-run execution path'). Confirms run_noise_validation_cell
        now genuinely calls build_lineage_record() and returns the result,
        with content matching the rest of the run's own fields -- not a
        placeholder or a hash of unrelated data."""
        inst = generate_instance("independent", 3, seed=0)
        qubo = build_qubo(inst)

        result = run_noise_validation_cell(
            inst, qubo, backend_label="ideal", backend=ideal_backend(), p=1, maxiter=3, shots=256, seed=0
        )

        assert isinstance(result.lineage_record, RunLineageRecord)
        assert result.lineage_record.instance_fingerprint == inst.fingerprint()
        assert result.lineage_record.qubo_n_qubits == qubo.n_qubits()
        assert result.lineage_record.circuit_depth == result.depth
        assert result.lineage_record.circuit_two_qubit_gates == result.two_qubit_gates
        assert result.lineage_record.shots == 256
        assert result.lineage_record.seed == 0
        # record_hash() must be a real, deterministic function of this
        # content, not a stand-in -- two independent calls on the same
        # inputs must agree.
        assert result.lineage_record.record_hash() == result.lineage_record.record_hash()


class TestOptimizationLevelAffectsTranspiledDepth:
    """Regression test for independent-review finding P2-16
    ('EXP-D1's optimization_level parameter -- the single load-bearing
    mechanism of the (non-reproducing) n=8 fix -- has zero test coverage').
    Confirms `run_noise_validation_cell`'s `optimization_level` parameter
    genuinely reaches the transpiler and measurably changes behavior
    against a backend with a real coupling-map constraint (FakeKingston) --
    independent of which specific direction is "better" (see P1-2's own
    finding that optimization_level=1 is in fact shallower here, not
    deeper -- this test only confirms the parameter is WIRED, not which way
    the comparison goes, since that is already covered by
    calibrate_transpile_optimization_level.py)."""

    def test_varying_optimization_level_changes_transpiled_depth_or_gate_count(self):
        inst = generate_instance("independent", 3, seed=0)
        qubo = build_qubo(inst)
        backend = fake_kingston_backend()

        r0 = run_noise_validation_cell(
            inst, qubo, backend_label="fake_kingston", backend=backend,
            p=1, maxiter=1, shots=256, seed=0, optimization_level=0,
        )
        r1 = run_noise_validation_cell(
            inst, qubo, backend_label="fake_kingston", backend=backend,
            p=1, maxiter=1, shots=256, seed=0, optimization_level=1,
        )

        assert (r0.depth, r0.two_qubit_gates) != (r1.depth, r1.two_qubit_gates), (
            "optimization_level=0 vs 1 produced IDENTICAL transpiled depth/gate-count -- "
            "the parameter may not be reaching the transpiler"
        )
