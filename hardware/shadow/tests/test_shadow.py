"""
Tests for Shadow Deployment.

Acceptance criteria:
- shadow_extractor: shadow_dataset returns correct keys and shapes
- save/load shadow_dataset roundtrip preserves Phi and y
- shadow_kernel: ShadowKernel fit+transform shapes are correct
- shadow_kernel inference_latency_estimate < 10ms for 5000 train samples
- shadow_deploy: package_vqc_shadow / package_pqk_shadow return correct keys
- vqc_shadow_infer: probabilities sum to 1
- _serialize_array / _deserialize_array roundtrip
- publish_shadow returns False when REDIS_URL absent
"""

import os
import tempfile

import numpy as np
import pytest


def simple_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
    """Simple RBF kernel for testing."""
    diff = x1 - x2
    return float(np.exp(-0.5 * np.dot(diff, diff)))


class TestShadowKernel:
    def _fit_kernel(self, m=10):
        from hardware.shadow.shadow_kernel import ShadowKernel
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        sk = ShadowKernel(m=m)
        sk.fit(X, simple_kernel, rng=rng)
        return sk, X

    def test_fit_transform_shape(self):
        from hardware.shadow.shadow_kernel import ShadowKernel
        sk, X = self._fit_kernel(m=10)
        Phi = sk.transform(X, simple_kernel)
        assert Phi.shape == (len(X), 10), (
            f"Expected (N, m)={(len(X), 10)}, got {Phi.shape}"
        )

    def test_landmarks_shape(self):
        sk, _ = self._fit_kernel(m=8)
        assert sk.landmarks_.shape[0] == 8

    def test_K_LL_inv_sqrt_shape(self):
        sk, _ = self._fit_kernel(m=8)
        assert sk.K_LL_inv_sqrt_.shape == (8, 8)

    def test_transform_before_fit_raises(self):
        from hardware.shadow.shadow_kernel import ShadowKernel
        sk = ShadowKernel(m=5)
        with pytest.raises(RuntimeError, match="fit"):
            sk.transform(np.zeros((3, 4)), simple_kernel)

    def test_predict_kernel_row_shape(self):
        sk, X = self._fit_kernel(m=10)
        Phi = sk.transform(X, simple_kernel)
        x_new = np.random.default_rng(99).standard_normal(5)
        K_row = sk.predict_kernel_row(x_new, Phi, simple_kernel)
        assert K_row.shape == (len(X),), (
            f"Expected kernel row shape ({len(X)},), got {K_row.shape}"
        )

    def test_inference_latency_target(self):
        from hardware.shadow.shadow_kernel import ShadowKernel
        sk = ShadowKernel(m=200)
        sk._fitted = True
        latency = sk.inference_latency_estimate_ms(5000)
        assert latency < 10.0, (
            f"Shadow kernel must achieve <10ms for 5000 samples, got {latency:.4f}ms"
        )

    def test_save_load_roundtrip(self):
        from hardware.shadow.shadow_kernel import ShadowKernel
        sk, X = self._fit_kernel(m=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sk.npz")
            sk.save(path)
            sk2 = ShadowKernel.load(path)
            np.testing.assert_allclose(sk.landmarks_, sk2.landmarks_, atol=1e-10)
            np.testing.assert_allclose(
                sk.K_LL_inv_sqrt_, sk2.K_LL_inv_sqrt_, atol=1e-10
            )

    def test_summary_latency_ok(self):
        from hardware.shadow.shadow_kernel import ShadowKernel
        sk = ShadowKernel(m=200)
        sk._fitted = True
        s = sk.summary()
        assert s["latency_ok"] is True, (
            "Shadow kernel with m=200 must achieve <10ms latency target"
        )

    def test_summary_keys(self):
        from hardware.shadow.shadow_kernel import ShadowKernel
        sk = ShadowKernel(m=10)
        s = sk.summary()
        assert "m_landmarks" in s
        assert "fitted" in s
        assert "inference_latency_ms_5k_train" in s


class TestShadowDeploy:
    def test_serialize_deserialize_roundtrip(self):
        from hardware.shadow.shadow_deploy import _serialize_array, _deserialize_array
        arr = np.random.default_rng(0).standard_normal((5, 3))
        serialized = _serialize_array(arr)
        recovered = _deserialize_array(serialized)
        np.testing.assert_allclose(arr, recovered, atol=1e-10)

    def test_package_vqc_shadow_keys(self):
        from hardware.shadow.shadow_deploy import package_vqc_shadow
        params = np.zeros(40)
        W = np.zeros((20, 3))
        b = np.zeros(3)
        payload = package_vqc_shadow(params, W, b, n_qubits=10, d_layers=2)
        for key in ["component", "params", "W", "b", "n_qubits", "d_layers", "timestamp"]:
            assert key in payload, f"Missing key: {key}"
        assert payload["component"] == "vqc_v5"

    def test_package_pqk_shadow_keys(self):
        from hardware.shadow.shadow_deploy import package_pqk_shadow
        landmarks = np.zeros((10, 5))
        K_inv = np.eye(10)
        W = np.zeros((10, 3))
        b = np.zeros(3)
        payload = package_pqk_shadow(landmarks, K_inv, W, b)
        for key in ["component", "landmarks", "K_LL_inv_sqrt", "W", "b"]:
            assert key in payload
        assert payload["component"] == "pqk_v2"

    def test_publish_shadow_false_without_redis(self):
        from hardware.shadow.shadow_deploy import publish_shadow
        old = os.environ.pop("REDIS_URL", None)
        try:
            result = publish_shadow({"component": "test", "timestamp": "now"})
            assert result is False
        finally:
            if old is not None:
                os.environ["REDIS_URL"] = old

    def test_vqc_shadow_infer_proba_sums_to_1(self):
        from hardware.shadow.shadow_deploy import package_vqc_shadow, vqc_shadow_infer

        rng = np.random.default_rng(42)
        n, d = 4, 2
        n_params = d * 2 * n
        params = rng.standard_normal(n_params) * 0.01
        W = rng.standard_normal((n, 3)) * 0.1
        b = rng.standard_normal(3) * 0.1
        payload = package_vqc_shadow(params, W, b, n_qubits=n, d_layers=d)

        x = rng.standard_normal(n)
        proba = vqc_shadow_infer(x, payload)
        assert proba.shape == (3,), f"Expected shape (3,), got {proba.shape}"
        assert abs(proba.sum() - 1.0) < 1e-6, (
            f"VQC shadow probabilities must sum to 1, got {proba.sum()}"
        )

    def test_vqc_shadow_infer_non_negative(self):
        from hardware.shadow.shadow_deploy import package_vqc_shadow, vqc_shadow_infer
        rng = np.random.default_rng(7)
        n, d = 4, 2
        params = rng.standard_normal(d * 2 * n) * 0.01
        W = rng.standard_normal((n, 3)) * 0.1
        b = np.zeros(3)
        payload = package_vqc_shadow(params, W, b, n_qubits=n, d_layers=d)
        x = rng.standard_normal(n)
        proba = vqc_shadow_infer(x, payload)
        assert np.all(proba >= 0), "Probabilities must be non-negative"

    def test_load_shadow_none_without_redis(self):
        from hardware.shadow.shadow_deploy import load_shadow
        old = os.environ.pop("REDIS_URL", None)
        try:
            result = load_shadow("vqc_v5")
            assert result is None
        finally:
            if old is not None:
                os.environ["REDIS_URL"] = old


class TestShadowExtractor:
    def test_shadow_dataset_required_keys(self):
        """shadow_dataset returns all required metadata keys."""
        from hardware.shadow.shadow_extractor import shadow_dataset
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 10))
        y = rng.integers(0, 3, 20)
        ds = shadow_dataset(X, y, component="qrc")
        for key in ["Phi", "y", "component", "timestamp", "n_samples",
                    "feature_dim", "data_hash"]:
            assert key in ds, f"Missing key: {key}"

    def test_shadow_dataset_shapes(self):
        from hardware.shadow.shadow_extractor import shadow_dataset
        rng = np.random.default_rng(42)
        N, F = 10, 8
        X = rng.standard_normal((N, F))
        y = rng.integers(0, 2, N)
        ds = shadow_dataset(X, y, component="qrc")
        assert ds["Phi"].shape[0] == N
        assert len(ds["y"]) == N
        assert ds["n_samples"] == N

    def test_shadow_dataset_unknown_component_raises(self):
        from hardware.shadow.shadow_extractor import shadow_dataset
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 5))
        y = rng.integers(0, 2, 5)
        with pytest.raises(ValueError, match="Unknown component"):
            shadow_dataset(X, y, component="xyz")

    def test_save_load_roundtrip(self):
        """save_shadow_dataset then load_shadow_dataset roundtrip."""
        from hardware.shadow.shadow_extractor import save_shadow_dataset, load_shadow_dataset
        rng = np.random.default_rng(42)
        Phi = rng.standard_normal((20, 5))
        y = rng.integers(0, 2, 20)
        dataset = {
            "Phi": Phi,
            "y": y,
            "component": "qrc",
            "timestamp": "2026-06-30T00:00:00+00:00",
            "n_samples": 20,
            "feature_dim": 5,
            "data_hash": "abc123",
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            save_shadow_dataset(dataset, path)
            loaded = load_shadow_dataset(path)
            np.testing.assert_allclose(loaded["Phi"], Phi, atol=1e-10)
            np.testing.assert_array_equal(loaded["y"], y)
            assert loaded["component"] == "qrc"
