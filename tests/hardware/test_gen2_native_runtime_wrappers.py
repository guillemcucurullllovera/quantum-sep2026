"""
Tests for Gen2's IBM Runtime V2 Wrappers (historical implementation).

Acceptance criteria:
- EMConfig.for_component returns correct EM stack per component matrix
- QRC: no ZNE, DD+TREX only
- PQK: ZNE + PEC flag + DD + twirl + TREX
- VQC: ZNE + DD + twirl + TREX (no PEC)
- QAOA: ZNE + DD + twirl + TREX (no PEC)
- EstimatorWrapper.run() raises RuntimeError when qiskit_ibm_runtime not installed
- SamplerWrapper.bitstrings_to_z_expectations gives values in [-1, 1]
- NoiseLearner: stale cache raises RuntimeError; uninit raises RuntimeError
"""

import numpy as np
import pytest

from hardware.runtime_versions.gen2_native.estimator_wrapper import (
    QuantumComponent,
    EMConfig,
    EstimatorWrapper,
)
from hardware.runtime_versions.gen2_native.sampler_wrapper import SamplerWrapper
import hardware.runtime_versions.gen2_native.noise_learner as nl


class TestEMConfig:
    def test_qrc_no_zne(self):
        cfg = EMConfig.for_component(QuantumComponent.QRC)
        assert cfg.zne_noise_factors == [], "QRC must not use ZNE"

    def test_qrc_dd_trex_only(self):
        cfg = EMConfig.for_component(QuantumComponent.QRC)
        assert cfg.use_dd is True
        assert cfg.use_trex is True
        assert cfg.use_pec is False
        assert cfg.use_twirl is False

    def test_pqk_pec_enabled(self):
        cfg = EMConfig.for_component(QuantumComponent.PQK)
        assert cfg.use_pec is True, "PQK v2 requires PEC for kernel accuracy"

    def test_pqk_full_stack(self):
        cfg = EMConfig.for_component(QuantumComponent.PQK)
        assert cfg.zne_noise_factors == [1, 2, 3]
        assert cfg.use_dd is True
        assert cfg.use_twirl is True
        assert cfg.n_twirl_circuits == 128
        assert cfg.use_trex is True

    def test_vqc_no_pec(self):
        cfg = EMConfig.for_component(QuantumComponent.VQC)
        assert cfg.use_pec is False

    def test_vqc_full_stack(self):
        cfg = EMConfig.for_component(QuantumComponent.VQC)
        assert cfg.zne_noise_factors == [1, 2, 3]
        assert cfg.use_dd is True
        assert cfg.use_twirl is True
        assert cfg.n_twirl_circuits == 128
        assert cfg.use_trex is True

    def test_qaoa_no_pec(self):
        cfg = EMConfig.for_component(QuantumComponent.QAOA)
        assert cfg.use_pec is False

    def test_qaoa_zne_dd_twirl(self):
        cfg = EMConfig.for_component(QuantumComponent.QAOA)
        assert cfg.zne_noise_factors == [1, 2, 3]
        assert cfg.use_dd is True
        assert cfg.use_twirl is True

    def test_component_enum_values(self):
        assert QuantumComponent.QRC.value == "qrc"
        assert QuantumComponent.PQK.value == "pqk_v2"
        assert QuantumComponent.VQC.value == "vqc_v5"
        assert QuantumComponent.QAOA.value == "qaoa_v2"

    def test_dd_sequence_default(self):
        cfg = EMConfig.for_component(QuantumComponent.VQC)
        assert cfg.dd_sequence == "XY4"

    def test_shots_default_positive(self):
        for comp in QuantumComponent:
            cfg = EMConfig.for_component(comp)
            assert cfg.shots > 0, f"shots must be positive for {comp}"


class TestEstimatorWrapper:
    def test_run_raises_without_qiskit(self):
        """When qiskit_ibm_runtime is absent, run() must raise RuntimeError."""
        wrapper = EstimatorWrapper(backend=None)
        assert wrapper._qiskit_estimator is None, (
            "qiskit_ibm_runtime is not installed in this env; "
            "_qiskit_estimator should be None"
        )
        with pytest.raises(RuntimeError, match="qiskit_ibm_runtime"):
            wrapper.run([])

    def test_for_component_factory(self):
        wrapper = EstimatorWrapper.for_component(QuantumComponent.VQC, backend=None)
        assert wrapper.em_config.component == QuantumComponent.VQC

    def test_em_summary_keys(self):
        wrapper = EstimatorWrapper.for_component(QuantumComponent.PQK, backend=None)
        summary = wrapper.em_summary()
        assert "component" in summary
        assert "zne" in summary
        assert "pec" in summary
        assert "trex" in summary

    def test_em_summary_pqk_pec(self):
        wrapper = EstimatorWrapper.for_component(QuantumComponent.PQK, backend=None)
        assert wrapper.em_summary()["pec"] is True

    def test_em_summary_qrc_no_zne(self):
        wrapper = EstimatorWrapper.for_component(QuantumComponent.QRC, backend=None)
        assert wrapper.em_summary()["zne"] is None


class TestSamplerWrapper:
    def test_run_raises_without_qiskit(self):
        wrapper = SamplerWrapper(backend=None)
        with pytest.raises(RuntimeError, match="qiskit_ibm_runtime"):
            wrapper.run([])

    def test_sample_bitstrings_shape(self):
        wrapper = SamplerWrapper(backend=None)
        n = 4
        sv = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
        counts = wrapper.sample_bitstrings(sv, n_shots=500, rng=np.random.default_rng(42))
        assert counts.shape == (2 ** n,)
        assert counts.sum() == 500

    def test_sample_bitstrings_sums_to_shots(self):
        wrapper = SamplerWrapper(backend=None)
        sv = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00>
        counts = wrapper.sample_bitstrings(sv, n_shots=100, rng=np.random.default_rng(0))
        assert counts.sum() == 100
        assert counts[0] == 100  # pure |00> always measures 0

    def test_z_expectations_range(self):
        wrapper = SamplerWrapper(backend=None)
        n = 4
        sv = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
        counts = wrapper.sample_bitstrings(sv, n_shots=1000, rng=np.random.default_rng(7))
        z = wrapper.bitstrings_to_z_expectations(counts, n=n, n_shots=1000)
        assert z.shape == (n,)
        assert np.all(z >= -1.0) and np.all(z <= 1.0), (
            f"Z-expectations must be in [-1, 1], got {z}"
        )

    def test_z_expectations_zero_state(self):
        """Pure |0>^n gives <Z_i> = +1 for all qubits."""
        wrapper = SamplerWrapper(backend=None)
        n = 3
        sv = np.zeros(2 ** n, dtype=complex)
        sv[0] = 1.0
        counts = wrapper.sample_bitstrings(sv, n_shots=200, rng=np.random.default_rng(0))
        z = wrapper.bitstrings_to_z_expectations(counts, n=n, n_shots=200)
        np.testing.assert_allclose(z, np.ones(n), atol=1e-10, err_msg=(
            "|0>^n should give Z-expectations all +1"
        ))

    def test_z_expectations_all_ones_state(self):
        """Pure |1>^n (index 2**n - 1) gives <Z_i> = -1 for all qubits."""
        wrapper = SamplerWrapper(backend=None)
        n = 3
        sv = np.zeros(2 ** n, dtype=complex)
        sv[-1] = 1.0
        counts = wrapper.sample_bitstrings(sv, n_shots=200, rng=np.random.default_rng(0))
        z = wrapper.bitstrings_to_z_expectations(counts, n=n, n_shots=200)
        np.testing.assert_allclose(z, -np.ones(n), atol=1e-10, err_msg=(
            "|1>^n should give Z-expectations all -1"
        ))


class TestNoiseLearnerProtocol:
    def setup_method(self):
        """Reset global cache before each test."""
        nl._CACHE = None

    def test_cache_status_uninitialized(self):
        status = nl.cache_status()
        assert status["status"] == "uninitialized"
        assert status["fresh"] is False
        assert status["age_minutes"] is None

    def test_assert_fresh_raises_when_uninitialized(self):
        with pytest.raises(RuntimeError, match="NoiseLearner result"):
            nl.assert_fresh_or_raise("PQK_v2")

    def test_assert_fresh_raises_when_stale(self):
        import time
        nl._CACHE = nl.NoiseLearnerCache(
            backend_name="test",
            noise_model=object(),
            timestamp_s=time.time() - nl.SESSION_TTL_SECONDS - 1,
        )
        with pytest.raises(RuntimeError, match="stale"):
            nl.assert_fresh_or_raise("PQK_v2")

    def test_assert_fresh_passes_when_fresh(self):
        import time
        nl._CACHE = nl.NoiseLearnerCache(
            backend_name="test",
            noise_model=object(),
            timestamp_s=time.time(),
        )
        nl.assert_fresh_or_raise("PQK_v2")  # Should not raise

    def test_session_ttl_is_2h(self):
        assert nl.SESSION_TTL_SECONDS == 7200, (
            "TTL must be 7200s (2h) to cover IBM Heron r3 T1 variation window"
        )

    def test_run_noise_learner_returns_none_without_qiskit(self):
        result = nl.run_noise_learner(backend=None)
        assert result is None, (
            "run_noise_learner should return None gracefully when qiskit absent"
        )
        assert nl._CACHE is None

    def test_ecr_noise_rate_documented(self):
        import time
        sentinel = object()
        nl._CACHE = nl.NoiseLearnerCache(
            backend_name="ibm_boston",
            noise_model=sentinel,
            timestamp_s=time.time(),
        )
        assert nl._CACHE.ecr_noise_rate == pytest.approx(0.024, abs=1e-6), (
            "ECR gate noise rate must be lambda~=0.024 for IBM Heron r3"
        )
