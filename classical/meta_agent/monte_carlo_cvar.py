"""
C1.5 — Monte Carlo CVaR with regime-conditional historical bootstrap.

The Bayesian gate (C1.4) decides HOW to allocate across strategies. This
component decides HOW MUCH to allocate at all — it produces a gross-
exposure scalar in [0, 1] that scales C1.4's weight vector so the
portfolio's monthly Conditional Value at Risk (CVaR) stays within a
target loss budget.

Final allocation at time t:
    w_final(t) = gross_exposure(t) * w_C14(t)
    cash(t)    = 1 - gross_exposure(t)

Method (regime-conditional historical bootstrap):

    At time t with regime probabilities π_t = (P_bull, P_bear, P_crisis):

    1. Compute the portfolio return that each historical month s would have
       produced under the proposed C1.4 weights w_t:
            r_p(s) = sum_k w_t^(k) * R(s, k)
       This converts the (T, K) historical return matrix into a (T,)
       vector of "what-if portfolio returns under today's weights".

    2. Compute a regime-similarity weight for each historical month:
            similarity(s) = sum_r  π_t(r) * Π(s, r)
       where Π is the (T, S) historical regime-probability matrix from C1.1.
       Months whose regime mixture resembled today's get higher weight.
       Bonus: when π_t is one-hot at crisis, this collapses to "weight each
       past month by its own P(crisis)" — exactly what we want for tail
       estimation.

    3. Bootstrap-resample N=10,000 portfolio returns from r_p using
       similarity(s) as the categorical sampling distribution. The result
       is a draw from the forward-looking portfolio return distribution
       conditional on regime context.

    4. Compute VaR_α and CVaR_α from the bootstrap draws:
            VaR_α  = (1-α)-quantile of the draws (a negative number)
            CVaR_α = mean of draws below VaR_α (a more negative number)

    5. Compute gross-exposure scalar:
            target = -target_loss_budget   (e.g. -0.05)
            if CVaR_α < target:
                gross = target / CVaR_α    (a positive number in (0, 1))
            else:
                gross = 1.0
       Capped at 1.0 (we never lever up) and floored at min_gross
       (we never fully de-risk, since some exposure is always required
       for the strategy to function).

Why bootstrap over parametric:
    The whole point of CVaR vs vol-target is to capture FAT TAILS. A
    Gaussian Monte Carlo would understate crisis risk by 3-5x because
    monthly returns have kurtosis ~5-10 (vs Gaussian's 3). Empirical
    bootstrap captures the real tail directly — March 2020 lives in
    the sample.

Why regime conditioning:
    Without it, the bootstrap weights all 240 historical months equally
    at all times, so the CVaR estimate is a slow-moving 20-year average.
    With regime conditioning, the CVaR estimate jumps when the HMM says
    "this looks like 2008" because the bootstrap then draws preferentially
    from 2008-2009 months.

No quantum dependencies. Pure Python + NumPy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import joblib


# ── data containers ──────────────────────────────────────────────

@dataclass
class CVaRSummary:
    """Human-readable summary of a fitted MonteCarloCVaR estimator."""
    alpha: float                  # confidence level (e.g. 0.95)
    target_loss: float            # loss budget (e.g. 0.05 = 5%)
    n_bootstrap: int
    max_gross: float
    min_gross: float
    n_history_months: int
    n_strategies: int
    n_regimes: int
    strategy_names: list
    regime_names: list

    def to_dict(self) -> dict:
        return asdict(self)


# ── main class ────────────────────────────────────────────────────

class MonteCarloCVaR:
    """
    Regime-conditional historical-bootstrap CVaR estimator and gross-
    exposure scaler.

    Standard usage:

        cvar = MonteCarloCVaR(alpha=0.95, target_loss=0.05, n_bootstrap=10000)
        cvar.fit(historical_returns, historical_regime_probs)

        # At each timestep:
        cvar_estimate = cvar.estimate_cvar(weights_t, regime_probs_t)
        gross = cvar.compute_gross_exposure(weights_t, regime_probs_t)
        scaled_weights = cvar.apply_to_weights(weights_t, regime_probs_t)
    """

    def __init__(
        self,
        alpha: float = 0.95,
        target_loss: float = 0.05,
        n_bootstrap: int = 10_000,
        max_gross: float = 1.0,
        min_gross: float = 0.05,
        random_state: int = 42,
    ):
        if not (0.5 <= alpha < 1.0):
            raise ValueError(f"alpha must be in [0.5, 1.0); got {alpha}")
        if target_loss <= 0.0:
            raise ValueError(f"target_loss must be > 0 (it is a magnitude); got {target_loss}")
        if n_bootstrap < 100:
            raise ValueError(f"n_bootstrap must be >= 100; got {n_bootstrap}")
        if not (0.0 <= min_gross <= max_gross <= 1.0):
            raise ValueError(
                f"need 0 <= min_gross <= max_gross <= 1; got min={min_gross}, max={max_gross}"
            )

        self.alpha = float(alpha)
        self.target_loss = float(target_loss)
        self.n_bootstrap = int(n_bootstrap)
        self.max_gross = float(max_gross)
        self.min_gross = float(min_gross)
        self.random_state = int(random_state)

        # Set by fit()
        self._returns: Optional[np.ndarray] = None         # (T, K)
        self._regime_probs: Optional[np.ndarray] = None    # (T, S)
        self._strategy_names: list = []
        self._regime_names: list = []
        self._fitted = False
        self._rng = np.random.default_rng(random_state)

    # ── core API ──

    def fit(
        self,
        returns: np.ndarray,
        regime_probs: np.ndarray,
        strategy_names: Optional[Sequence[str]] = None,
        regime_names: Optional[Sequence[str]] = None,
    ) -> "MonteCarloCVaR":
        """
        Store historical returns + regime probabilities. No learning step —
        bootstrap CVaR is non-parametric.

        Args:
            returns:      (T, K) historical per-strategy returns
            regime_probs: (T, S) historical regime probabilities, rows ≈ 1
        """
        returns = np.asarray(returns, dtype=np.float64)
        regime_probs = np.asarray(regime_probs, dtype=np.float64)
        if returns.ndim != 2:
            raise ValueError(f"returns must be 2-D (T, K); got {returns.shape}")
        if regime_probs.ndim != 2:
            raise ValueError(f"regime_probs must be 2-D (T, S); got {regime_probs.shape}")
        if returns.shape[0] != regime_probs.shape[0]:
            raise ValueError(
                f"T mismatch: returns T={returns.shape[0]}, "
                f"regime_probs T={regime_probs.shape[0]}"
            )
        if not np.isfinite(returns).all():
            raise ValueError("returns contains NaN or inf")
        if not np.isfinite(regime_probs).all() or (regime_probs < -1e-9).any():
            raise ValueError("regime_probs must be non-negative finite")

        self._returns = returns
        self._regime_probs = regime_probs
        K = returns.shape[1]
        S = regime_probs.shape[1]
        self._strategy_names = (
            list(strategy_names) if strategy_names is not None
            else [f"strategy_{i}" for i in range(K)]
        )
        self._regime_names = (
            list(regime_names) if regime_names is not None
            else (["bull", "bear", "crisis"] if S == 3 else [f"regime_{i}" for i in range(S)])
        )
        self._fitted = True
        return self

    def _bootstrap_portfolio_returns(
        self,
        weights: np.ndarray,
        regime_probs_t: np.ndarray,
        n_bootstrap: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw `n_bootstrap` portfolio-return samples using regime-similarity
        weighting. Returns a 1-D array of bootstrapped portfolio returns.
        """
        n_bootstrap = n_bootstrap or self.n_bootstrap
        T = self._returns.shape[0]

        # Step 1: portfolio return at each historical month under proposed weights
        portfolio_returns_hist = self._returns @ weights              # (T,)

        # Step 2: regime-similarity weights
        # similarity(s) = sum_r  π_t(r) * Π(s, r)
        similarity = self._regime_probs @ regime_probs_t              # (T,)
        sim_sum = similarity.sum()
        if sim_sum <= 0:
            # Degenerate: all-zero similarity → fall back to uniform
            sample_p = np.full(T, 1.0 / T)
        else:
            sample_p = similarity / sim_sum

        # Step 3: bootstrap-resample
        idx = self._rng.choice(T, size=n_bootstrap, p=sample_p, replace=True)
        return portfolio_returns_hist[idx]

    def estimate_cvar(
        self,
        weights: np.ndarray,
        regime_probs_t: np.ndarray,
        n_bootstrap: Optional[int] = None,
    ) -> dict:
        """
        Estimate VaR_α and CVaR_α for the proposed weights given current
        regime probabilities.

        Returns a dict with keys: var, cvar, mean, std, n_samples.
        var and cvar are NEGATIVE numbers representing losses
        (e.g. cvar = -0.07 means expected loss in worst 5% is 7%).
        """
        self._check_fitted()
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != self._returns.shape[1]:
            raise ValueError(
                f"weights has K={weights.shape[0]}, "
                f"history has K={self._returns.shape[1]}"
            )
        regime_probs_t = np.asarray(regime_probs_t, dtype=np.float64).reshape(-1)
        if regime_probs_t.shape[0] != self._regime_probs.shape[1]:
            raise ValueError(
                f"regime_probs_t has S={regime_probs_t.shape[0]}, "
                f"history has S={self._regime_probs.shape[1]}"
            )

        draws = self._bootstrap_portfolio_returns(weights, regime_probs_t, n_bootstrap)
        var_q = np.quantile(draws, 1.0 - self.alpha)         # e.g. 5th percentile
        tail = draws[draws <= var_q]
        cvar = float(tail.mean()) if tail.size > 0 else float(var_q)
        return {
            "var": float(var_q),
            "cvar": cvar,
            "mean": float(draws.mean()),
            "std": float(draws.std()),
            "n_samples": int(draws.size),
        }

    def compute_gross_exposure(
        self,
        weights: np.ndarray,
        regime_probs_t: np.ndarray,
        n_bootstrap: Optional[int] = None,
    ) -> float:
        """
        Return the gross-exposure scalar in [min_gross, max_gross].

        Logic:
            if CVaR is more negative than -target_loss:
                gross = -target_loss / CVaR     (a number in (0, 1))
            else (CVaR is within budget):
                gross = max_gross               (typically 1.0)
        """
        est = self.estimate_cvar(weights, regime_probs_t, n_bootstrap)
        cvar = est["cvar"]
        target = -self.target_loss

        if cvar < target and cvar < 0:
            # Need to scale down so the linearly-scaled CVaR hits target
            gross = target / cvar              # in (0, 1)
        else:
            gross = self.max_gross

        # Clip to bounds
        gross = float(np.clip(gross, self.min_gross, self.max_gross))
        return gross

    def apply_to_weights(
        self,
        weights: np.ndarray,
        regime_probs_t: np.ndarray,
        n_bootstrap: Optional[int] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Apply gross-exposure scaling to a C1.4 weight vector.

        Returns: (scaled_weights, gross_exposure)
            scaled_weights: original weights * gross_exposure (sums to gross)
            gross_exposure: scalar in [min_gross, max_gross]

        The cash position is 1 - gross_exposure.
        """
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        gross = self.compute_gross_exposure(weights, regime_probs_t, n_bootstrap)
        return weights * gross, gross

    def apply_batch(
        self,
        weights_T: np.ndarray,
        regime_probs_T: np.ndarray,
        n_bootstrap: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized application over a time series.

        Args:
            weights_T:      (T, K) per-timestep C1.4 weights
            regime_probs_T: (T, S) per-timestep regime probabilities

        Returns:
            scaled_weights: (T, K)
            gross_series:   (T,)
        """
        weights_T = np.asarray(weights_T, dtype=np.float64)
        regime_probs_T = np.asarray(regime_probs_T, dtype=np.float64)
        if weights_T.shape[0] != regime_probs_T.shape[0]:
            raise ValueError(
                f"T mismatch: weights {weights_T.shape[0]} vs regime_probs {regime_probs_T.shape[0]}"
            )
        T = weights_T.shape[0]
        scaled = np.empty_like(weights_T)
        gross = np.empty(T, dtype=np.float64)
        for t in range(T):
            scaled[t], gross[t] = self.apply_to_weights(
                weights_T[t], regime_probs_T[t], n_bootstrap=n_bootstrap
            )
        return scaled, gross

    # ── inspection ──

    def summary(self) -> CVaRSummary:
        """Produce a JSON-serializable summary."""
        self._check_fitted()
        return CVaRSummary(
            alpha=self.alpha,
            target_loss=self.target_loss,
            n_bootstrap=self.n_bootstrap,
            max_gross=self.max_gross,
            min_gross=self.min_gross,
            n_history_months=self._returns.shape[0],
            n_strategies=self._returns.shape[1],
            n_regimes=self._regime_probs.shape[1],
            strategy_names=list(self._strategy_names),
            regime_names=list(self._regime_names),
        )

    # ── persistence ──

    def save(self, path: str | Path) -> None:
        """Pickle the fitted estimator + history. Companion JSON summary."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "target_loss": self.target_loss,
            "n_bootstrap": self.n_bootstrap,
            "max_gross": self.max_gross,
            "min_gross": self.min_gross,
            "random_state": self.random_state,
            "returns": self._returns,
            "regime_probs": self._regime_probs,
            "strategy_names": self._strategy_names,
            "regime_names": self._regime_names,
        }
        joblib.dump(payload, path)
        summary_path = path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(self.summary().to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "MonteCarloCVaR":
        """Load a previously-saved MonteCarloCVaR."""
        payload = joblib.load(Path(path))
        inst = cls(
            alpha=payload["alpha"],
            target_loss=payload["target_loss"],
            n_bootstrap=payload["n_bootstrap"],
            max_gross=payload["max_gross"],
            min_gross=payload["min_gross"],
            random_state=payload["random_state"],
        )
        inst._returns = payload["returns"]
        inst._regime_probs = payload["regime_probs"]
        inst._strategy_names = payload["strategy_names"]
        inst._regime_names = payload["regime_names"]
        inst._fitted = True
        return inst

    # ── internals ──

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "MonteCarloCVaR has not been fitted; call .fit(returns, regime_probs) first"
            )


__all__ = ["MonteCarloCVaR", "CVaRSummary"]
