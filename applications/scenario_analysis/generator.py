"""`ScenarioGenerator`: real, distinct parametrized shocks (vol spike,
correlation breakdown, mean shift, jump) applied to a base P&L/return
sample or a multi-asset returns matrix -- so any downstream risk pipeline
(tail-risk, options, portfolio) can be re-evaluated under a genuinely
stressed distribution instead of the base one, not a relabeled copy of it.
"""
from __future__ import annotations

import numpy as np

from applications.scenario_analysis.domain import ScenarioSpec, ShockKind, ShockSpec

__all__ = ["ScenarioGenerator", "default_scenario_grid"]


class ScenarioGenerator:
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    # ── individual shocks ────────────────────────────────────────────

    def vol_spike(self, base: np.ndarray, multiplier: float) -> np.ndarray:
        """Scale deviations from the sample mean by `multiplier` -- a vol
        regime shift that preserves the mean but widens (>1) or narrows
        (<1) the spread."""
        base = np.asarray(base, dtype=float)
        mu = base.mean(axis=0)
        return mu + (base - mu) * float(multiplier)

    def mean_shift(self, base: np.ndarray, shift: float) -> np.ndarray:
        """Additive drift shock -- e.g. a persistent adverse-return regime."""
        return np.asarray(base, dtype=float) + float(shift)

    def jump(
        self, base: np.ndarray, *, jump_prob: float, jump_size: float, seed: int | None = None,
    ) -> np.ndarray:
        """Inject a discrete jump into a `jump_prob` fraction of samples,
        sized `jump_size` (typically negative -- a crash jump). A real
        structural break applied to a random subset, not a relabeling of
        the base sample."""
        if not (0.0 <= jump_prob <= 1.0):
            raise ValueError(f"jump_prob must be in [0, 1]; got {jump_prob}")
        base = np.asarray(base, dtype=float).copy()
        rng = np.random.default_rng(seed if seed is not None else self._seed)
        mask = rng.random(base.shape[0]) < jump_prob
        base[mask] = base[mask] + jump_size
        return base

    def correlation_breakdown(
        self, returns_matrix: np.ndarray, target_corr: float, *, seed: int | None = None,
    ) -> np.ndarray:
        """Force a multi-asset (T, K) returns matrix toward a target pairwise
        correlation (typically higher, modeling a crisis 'everything
        correlates to 1' regime) via a one-factor construction:

            z_k = sqrt(rho) * f + sqrt(1 - rho) * z_k'

        where f is a shared standard-normal factor and z_k' is asset k's own
        standardized idiosyncratic shock, rescaled back to each asset's
        original mean/std. A genuine resampling, not a covariance-matrix
        cosmetic edit.
        """
        returns_matrix = np.asarray(returns_matrix, dtype=float)
        if returns_matrix.ndim != 2:
            raise ValueError("correlation_breakdown requires a 2-D (T, K) returns matrix")
        if not (0.0 <= target_corr <= 1.0):
            raise ValueError(f"target_corr must be in [0, 1]; got {target_corr}")
        t, _k = returns_matrix.shape
        rng = np.random.default_rng(seed if seed is not None else self._seed)
        mu = returns_matrix.mean(axis=0)
        sigma = returns_matrix.std(axis=0)
        sigma_safe = np.where(sigma > 0, sigma, 1.0)
        z = (returns_matrix - mu) / sigma_safe
        common_factor = rng.standard_normal(t)
        blended = np.sqrt(target_corr) * common_factor[:, None] + np.sqrt(1.0 - target_corr) * z
        return blended * sigma_safe + mu

    # ── scenario application ─────────────────────────────────────────

    def apply_scenario(self, base: np.ndarray, scenario: ScenarioSpec, *, seed: int | None = None) -> np.ndarray:
        """Apply every shock in `scenario` sequentially to `base`."""
        out = np.asarray(base, dtype=float).copy()
        for shock in scenario.shocks:
            out = self._apply_shock(out, shock, seed=seed)
        return out

    def _apply_shock(self, series: np.ndarray, shock: ShockSpec, *, seed: int | None) -> np.ndarray:
        if shock.kind == ShockKind.VOL_SPIKE:
            return self.vol_spike(series, multiplier=shock.params.get("multiplier", 1.5))
        if shock.kind == ShockKind.MEAN_SHIFT:
            return self.mean_shift(series, shift=shock.params.get("shift", 0.0))
        if shock.kind == ShockKind.JUMP:
            return self.jump(
                series,
                jump_prob=shock.params.get("jump_prob", 0.02),
                jump_size=shock.params.get("jump_size", -0.1),
                seed=shock.params.get("seed", seed),
            )
        if shock.kind == ShockKind.CORRELATION_BREAKDOWN:
            if series.ndim != 2:
                raise ValueError(
                    "correlation_breakdown requires a 2-D (T, K) returns matrix -- "
                    "not applicable to a 1-D P&L series"
                )
            return self.correlation_breakdown(
                series, target_corr=shock.params.get("target_corr", 0.9), seed=shock.params.get("seed", seed),
            )
        raise ValueError(f"unhandled shock kind: {shock.kind}")  # pragma: no cover - exhaustive enum

    def generate_grid(
        self, base: np.ndarray, scenarios: list[ScenarioSpec], *, seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Apply each scenario to `base` and return {scenario_name: shocked_series}."""
        return {s.name: self.apply_scenario(base, s, seed=seed) for s in scenarios}


def default_scenario_grid() -> list[ScenarioSpec]:
    """A canonical, reusable scenario grid spanning all four shock families
    -- the platform's default cross-application stress grid."""
    return [
        ScenarioSpec(name="baseline", shocks=[]),
        ScenarioSpec(name="mild_vol_spike", shocks=[ShockSpec(kind=ShockKind.VOL_SPIKE, params={"multiplier": 1.5})]),
        ScenarioSpec(name="severe_vol_spike", shocks=[ShockSpec(kind=ShockKind.VOL_SPIKE, params={"multiplier": 3.0})]),
        ScenarioSpec(
            name="mean_shift_down", shocks=[ShockSpec(kind=ShockKind.MEAN_SHIFT, params={"shift": -0.02})],
        ),
        ScenarioSpec(
            name="crash_jump",
            shocks=[ShockSpec(kind=ShockKind.JUMP, params={"jump_prob": 0.03, "jump_size": -0.15})],
        ),
        ScenarioSpec(
            name="vol_spike_and_jump",
            shocks=[
                ShockSpec(kind=ShockKind.VOL_SPIKE, params={"multiplier": 2.0}),
                ShockSpec(kind=ShockKind.JUMP, params={"jump_prob": 0.02, "jump_size": -0.1}),
            ],
        ),
    ]
