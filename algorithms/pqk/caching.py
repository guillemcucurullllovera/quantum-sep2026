"""Content-addressed kernel-matrix cache.

Fidelity/projected Gram matrix computation is O(N^2) circuit evaluations --
expensive enough that re-running the same (dataset, feature map, params)
combination during a hyperparameter sweep or a notebook re-run should hit a
cache, not recompute from scratch. Keyed by a hash of the exact inputs, so a
cache hit is only ever returned for genuinely identical inputs.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from core.config import get_settings


def _hash_key(X: np.ndarray, feature_map_name: str, method: str, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(feature_map_name.encode())
    h.update(method.encode())
    h.update(extra.encode())
    h.update(np.ascontiguousarray(X).tobytes())
    return h.hexdigest()[:32]


class KernelCache:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (get_settings().data_dir / "pqk_kernel_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    def get_or_compute(
        self,
        X: np.ndarray,
        feature_map_name: str,
        method: str,
        compute_fn,
        extra: str = "",
    ) -> np.ndarray:
        """`compute_fn` is a zero-arg callable returning the Gram matrix if
        this is a cache miss. Returns the (possibly cached) matrix."""
        key = _hash_key(X, feature_map_name, method, extra)
        path = self._path(key)
        if path.exists():
            self.hits += 1
            with np.load(path) as data:
                return data["K"]

        self.misses += 1
        K = compute_fn()
        np.savez_compressed(path, K=K, computed_at=time.time())
        return K

    def clear(self) -> int:
        n = 0
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()
            n += 1
        self.hits = self.misses = 0
        return n

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
