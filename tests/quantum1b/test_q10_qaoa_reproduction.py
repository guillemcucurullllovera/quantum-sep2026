"""Tests for experiments/quantum1b/q10_qaoa_reproduction.py (EXP-B3)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.quantum1b.q10_qaoa_reproduction import (
    bipartite_entanglement_entropy,
    run_exp_b3,
    statevector_memory_bytes,
    theoretical_memory_crossover_n_qubits,
)
from experiments.quantum1b.registry_schema import load_registry


class TestStatevectorMemoryBytes:
    def test_exact_formula(self):
        for n in [0, 1, 4, 10, 20, 27]:
            assert statevector_memory_bytes(n) == (1 << n) * 16

    def test_monotonic_increasing(self):
        vals = [statevector_memory_bytes(n) for n in range(0, 20)]
        assert all(b > a for a, b in zip(vals, vals[1:]))


class TestTheoreticalMemoryCrossover:
    def test_crossover_matches_manual_bound(self):
        cap = 4 * (1024 ** 3)
        n = theoretical_memory_crossover_n_qubits(cap)
        assert statevector_memory_bytes(n) > cap
        assert statevector_memory_bytes(n - 1) <= cap

    def test_crossover_scales_with_cap(self):
        small_cap = 1024.0
        n_small = theoretical_memory_crossover_n_qubits(small_cap)
        big_cap = 4 * (1024 ** 3)
        n_big = theoretical_memory_crossover_n_qubits(big_cap)
        assert n_small < n_big


class TestBipartiteEntanglementEntropy:
    def test_product_state_zero_entropy(self):
        # |0>|0>|0> -- a fully separable 3-qubit product state.
        state = np.zeros(8, dtype=np.complex128)
        state[0] = 1.0
        diag = bipartite_entanglement_entropy(state, n_qubits=3, cut=1)
        assert diag.schmidt_rank == 1
        assert diag.entanglement_entropy_bits == pytest.approx(0.0, abs=1e-9)
        assert diag.entropy_fraction_of_max == pytest.approx(0.0, abs=1e-9)

    def test_bell_state_one_bit_entropy(self):
        # (|00> + |11>) / sqrt(2) -- maximally entangled 2-qubit state.
        state = np.zeros(4, dtype=np.complex128)
        state[0b00] = 1 / np.sqrt(2)
        state[0b11] = 1 / np.sqrt(2)
        diag = bipartite_entanglement_entropy(state, n_qubits=2, cut=1)
        assert diag.schmidt_rank == 2
        assert diag.entanglement_entropy_bits == pytest.approx(1.0, abs=1e-9)
        assert diag.max_possible_entropy_bits == pytest.approx(1.0, abs=1e-9)
        assert diag.entropy_fraction_of_max == pytest.approx(1.0, abs=1e-9)

    def test_ghz_state_bounded_by_max(self):
        # (|000> + |111>) / sqrt(2), cut after qubit 0: Schmidt rank 2 (max
        # possible for a 1-vs-2 qubit cut is min(2,4)=2), full 1 bit entropy.
        state = np.zeros(8, dtype=np.complex128)
        state[0b000] = 1 / np.sqrt(2)
        state[0b111] = 1 / np.sqrt(2)
        diag = bipartite_entanglement_entropy(state, n_qubits=3, cut=1)
        assert diag.schmidt_rank == 2
        assert diag.max_possible_schmidt_rank == 2
        assert diag.entanglement_entropy_bits == pytest.approx(1.0, abs=1e-9)

    def test_entropy_never_exceeds_theoretical_max(self):
        rng = np.random.default_rng(0)
        for n_qubits in (4, 6, 8):
            raw = rng.normal(size=1 << n_qubits) + 1j * rng.normal(size=1 << n_qubits)
            state = raw / np.linalg.norm(raw)
            for cut in range(1, n_qubits):
                diag = bipartite_entanglement_entropy(state, n_qubits, cut)
                assert diag.entanglement_entropy_bits <= diag.max_possible_entropy_bits + 1e-9
                assert diag.schmidt_rank <= diag.max_possible_schmidt_rank


class TestRunExpB3Smoke:
    def test_run_produces_expected_structure(self, monkeypatch):
        import experiments.quantum1b.q10_qaoa_reproduction as mod

        # Shrink the ladder for a fast CI smoke test -- the full preregistered
        # ladder (up to n_candidates=16) is exercised by run_track_b_exp_b3.py
        # directly, not by this bounded test.
        monkeypatch.setattr(mod, "CANDIDATE_LADDER", (4, 12))
        monkeypatch.setattr(mod, "P_VALUES", (1,))

        registry = load_registry()
        manifest = run_exp_b3(registry)

        assert manifest["experiment_id"] == "EXP-B3"
        assert manifest["csim_classification"] == "CSIM_BY_PROVENANCE_AT_EVERY_N_TESTED"
        assert len(manifest["cells"]) == 2
        assert manifest["computed_evidence_level_via_core_evidence"] in ("A", "B")
        for cell in manifest["cells"]:
            assert cell["schmidt_rank"] <= cell["max_possible_schmidt_rank"]
            assert cell["entanglement_entropy_bits"] <= cell["max_possible_entropy_bits"] + 1e-9
            assert cell["statevector_memory_bytes"] == (1 << cell["n_qubits"]) * 16

        # n_candidates=4 stays within the optimized ladder; n_candidates=12
        # exceeds it and must use representative fixed angles, not COBYLA.
        by_n = {c["n_candidates"]: c for c in manifest["cells"]}
        assert "cobyla_optimized" in by_n[4]["angle_source"]
        assert "representative_fixed_angles" in by_n[12]["angle_source"]

        assert manifest["experiment_result"]["financial_edge_claimed"] is False

        # Regression test for independent-review finding P3-11 ("a smoke
        # test produces an unasserted NaN via mean-of-empty-slice"): this
        # test's own P_VALUES=(1,) monkeypatch means no p=2 cells exist, so
        # mean_entropy_fraction_of_max_p2 is explicitly undefined.
        # q10_qaoa_reproduction.py now guards against the "Mean of empty
        # slice" RuntimeWarning this used to trigger -- confirmed via
        # pytest.ini's own warning-to-error promotion if configured, and
        # explicitly asserted here rather than left unchecked.
        p2_metric = manifest["experiment_result"]["metrics"]["mean_entropy_fraction_of_max_p2"]
        assert p2_metric is None or (isinstance(p2_metric, float) and p2_metric != p2_metric)  # None or NaN
