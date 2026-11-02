"""
C1.1 — Regime-detection HMM.

Three-state Gaussian-emission HMM (bull / bear / crisis) over a small macro
feature panel (VIX, term spread, DXY momentum, IG credit spread,
copper/gold ratio). The transition matrix carries a Dirichlet prior fit
against NBER expansion/recession labels so the unsupervised EM doesn't
flicker — a documented failure mode of HMMs on ~300 monthly observations.

Design choices and their rationale are documented in
`phase1/results/c1_1_regime_hmm/CALIBRATION_REPORT.md` (produced after the
real-data fit). This file is a thin wrapper over `hmmlearn.hmm.GaussianHMM`
that adds three things absent from the upstream library:

1. `build_nber_prior` — convert USREC (NBER binary recession label) into a
   3×3 Dirichlet alpha matrix using empirical expansion/recession dwell
   times. The alpha is then passed to hmmlearn as `transmat_prior`.

2. `_relabel_by_vix` — post-hoc state permutation so that state 0 is always
   bull (lowest mean VIX), state 1 is always bear (middle), state 2 is
   always crisis (highest). Necessary because EM produces unordered labels;
   without this the downstream Bayesian gate would see different state IDs
   across re-fits.

3. `RegimeHMM` class — fit/predict/save/load with the relabeling baked in,
   plus a `summary()` method that produces the per-state stationary
   distribution, mean dwell time, and conditional feature means that the
   calibration report consumes.

Pure-Python; no quantum dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# hmmlearn is a hard dependency; the import error is intentional if missing
# because there is no graceful degradation path for a model that won't fit.
from hmmlearn.hmm import GaussianHMM
import joblib


# ── NBER prior helper ──────────────────────────────────────────────

# Historical empirics, NBER Business Cycle Dating Committee:
#   median expansion length (post-1945):  58 months
#   median recession length:              11 months
# Source: https://www.nber.org/research/business-cycle-dating
NBER_MEDIAN_EXPANSION_MONTHS = 58.0
NBER_MEDIAN_RECESSION_MONTHS = 11.0


def build_nber_prior(
    usrec: Optional[np.ndarray] = None,
    n_states: int = 3,
    effective_n: float = 100.0,
    crisis_share_of_recession: float = 0.3,
) -> np.ndarray:
    """
    Build a Dirichlet-alpha transition prior for a 3-state HMM using NBER
    expansion/recession dwell-time empirics.

    The 3-state model maps to NBER as:
        state 0 (bull):    NBER expansion
        state 1 (bear):    NBER recession, ordinary
        state 2 (crisis):  NBER recession, extreme tail (top ~30% of recession
                           months by realized vol, conceptually)

    For each origin state we set diagonal stickiness from the implied dwell
    time, then distribute the remaining (1 - stickiness) mass to the other
    two states with a small bias toward the more "similar" neighboring state
    (bear ↔ crisis are adjacent; bull is far from crisis).

    Args:
        usrec: optional (T,) array of {0, 1} NBER recession indicators. If
               provided, the diagonal stickiness is calibrated empirically
               from this series rather than from the published median dwell
               times. Pass None to use the median-based default.
        n_states: must be 3 (this prior is purpose-built for the 3-state
                  bull/bear/crisis schema). Listed as an arg for clarity.
        effective_n: pseudo-count scale for the Dirichlet. Larger values
                     anchor the posterior more strongly to the prior;
                     ~100 is a mild prior on ~300 observations.
        crisis_share_of_recession: what fraction of recession-state dwell
                                   is attributed to the "crisis" sub-state
                                   rather than ordinary "bear". 0.3 means
                                   crisis is ~30% of recession months.

    Returns:
        alpha: (3, 3) Dirichlet alpha matrix, all entries > 0, suitable for
               hmmlearn's `transmat_prior` argument.
    """
    if n_states != 3:
        raise NotImplementedError(
            "build_nber_prior is purpose-built for 3-state bull/bear/crisis; "
            f"got n_states={n_states}"
        )

    # Empirical stickiness from USREC if provided, else medians from the literature
    if usrec is not None:
        usrec = np.asarray(usrec).astype(int)
        if set(np.unique(usrec).tolist()) - {0, 1}:
            raise ValueError("usrec must be binary {0, 1}")
        # P(stay | expansion) = (#months in expansion that stay in expansion) / (#expansion months)
        in_exp = usrec[:-1] == 0
        in_rec = usrec[:-1] == 1
        stay_exp = ((usrec[:-1] == 0) & (usrec[1:] == 0)).sum() / max(in_exp.sum(), 1)
        stay_rec = ((usrec[:-1] == 1) & (usrec[1:] == 1)).sum() / max(in_rec.sum(), 1)
    else:
        # Geometric distribution: P(stay) = 1 - 1/mean_dwell
        stay_exp = 1.0 - 1.0 / NBER_MEDIAN_EXPANSION_MONTHS  # ≈ 0.983
        stay_rec = 1.0 - 1.0 / NBER_MEDIAN_RECESSION_MONTHS  # ≈ 0.909

    # Within-recession allocation: bear vs crisis
    # P(bear | recession) = 1 - crisis_share, P(crisis | recession) = crisis_share
    bear_share = 1.0 - crisis_share_of_recession

    # Construct the 3×3 transition matrix (rows sum to 1)
    P = np.zeros((3, 3), dtype=np.float64)

    # Row 0 (bull):   stay_exp on diagonal; the (1 - stay_exp) leaves expansion,
    #                 split between bear (most likely next state) and crisis (rare).
    leave_bull = 1.0 - stay_exp
    P[0, 0] = stay_exp
    P[0, 1] = leave_bull * bear_share
    P[0, 2] = leave_bull * crisis_share_of_recession

    # Row 1 (bear):   stay_rec on diagonal (treating bear as the "ordinary recession" half).
    #                 Leaves split between bull (recovery) and crisis (escalation).
    #                 Recoveries are more common than escalations: 70/30.
    leave_bear = 1.0 - stay_rec
    P[1, 0] = leave_bear * 0.70
    P[1, 1] = stay_rec
    P[1, 2] = leave_bear * 0.30

    # Row 2 (crisis): less sticky than bear (crises de-escalate to bear, not directly to bull).
    #                 Use stay_rec - 0.05 as a heuristic for crisis stickiness.
    stay_crisis = max(stay_rec - 0.05, 0.5)
    leave_crisis = 1.0 - stay_crisis
    P[2, 0] = leave_crisis * 0.10                # rarely jump straight crisis → bull
    P[2, 1] = leave_crisis * 0.90                # usually crisis → bear
    P[2, 2] = stay_crisis

    # Convert to Dirichlet alpha by multiplying by effective sample size.
    # hmmlearn adds these to the row sums in the M-step, so an effective_n
    # of 100 means "as if we had 100 prior pseudo-transitions per row".
    alpha = P * effective_n

    # Floor to avoid zero alphas (Dirichlet is undefined at 0)
    alpha = np.maximum(alpha, 1e-3)
    return alpha


# ── post-hoc state labeling ────────────────────────────────────────

def _relabel_by_feature(
    model: GaussianHMM,
    feature_index: int = 0,
    ascending: bool = True,
) -> np.ndarray:
    """
    Compute the permutation that reorders the model's hidden states by the
    mean of one feature (default: feature 0, intended to be VIX). The
    returned permutation maps old_state_id → new_state_id such that
    new_state 0 has the smallest (or largest, if ascending=False) mean.

    The permutation is *applied* in-place: model.means_, model.covars_,
    model.transmat_, model.startprob_ are all reordered.
    """
    means = model.means_[:, feature_index]
    order = np.argsort(means) if ascending else np.argsort(-means)
    # `order[new_state] = old_state`. Build inverse for clarity.
    inv = np.argsort(order)   # `inv[old_state] = new_state`

    # Permute parameters
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[order][:, order]
    model.means_ = model.means_[order]
    # covars_ getter returns the expanded (n, d, d) form even for 'diag', but
    # the public setter validates against the canonical form per covariance
    # type. Operating on the private `_covars_` avoids the round-trip
    # mis-shape and is the form actually used by hmmlearn internally.
    if model.covariance_type == "tied":
        pass  # single covariance shared across states; no permutation needed
    else:
        model._covars_ = model._covars_[order]
    return inv


# ── data container ────────────────────────────────────────────────

@dataclass
class RegimeSummary:
    """Human-readable summary of a fitted RegimeHMM."""
    state_labels: list             # ["bull", "bear", "crisis"]
    stationary_distribution: list  # P(s) under the stationary distribution
    mean_dwell_months: list        # 1 / (1 - P(stay)) per state
    state_means: list              # list of per-state feature mean vectors
    transmat: list                 # 3×3 transition matrix
    n_iter: int                    # EM iterations performed
    converged: bool                # did EM converge within tol?
    feature_names: list

    def to_dict(self) -> dict:
        return asdict(self)


# ── main class ─────────────────────────────────────────────────────

DEFAULT_STATE_LABELS = ["bull", "bear", "crisis"]


class RegimeHMM:
    """
    Three-state Gaussian-emission HMM with NBER-informed transition prior
    and stable state labeling.

    Standard usage:

        prior = build_nber_prior(usrec=usrec_monthly)
        m = RegimeHMM(n_states=3, vix_col=0, random_state=42)
        m.fit(X_train, transmat_prior=prior)
        probs = m.predict_proba(X_test)   # (T, 3) in bull/bear/crisis order
    """

    def __init__(
        self,
        n_states: int = 3,
        vix_col: int = 0,
        covariance_type: str = "diag",
        n_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
        feature_names: Optional[Sequence[str]] = None,
    ):
        if n_states != 3:
            raise NotImplementedError(
                "RegimeHMM is purpose-built for 3 states (bull/bear/crisis); "
                f"got n_states={n_states}"
            )
        self.n_states = n_states
        self.vix_col = vix_col
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.feature_names = list(feature_names) if feature_names else None

        self._model: Optional[GaussianHMM] = None
        self._fitted = False

    # ── core API ──

    def fit(
        self,
        X: np.ndarray,
        transmat_prior: Optional[np.ndarray] = None,
        startprob_prior: Optional[np.ndarray] = None,
    ) -> "RegimeHMM":
        """
        Fit the HMM via EM. `X` is (T, n_features) — feature order must
        match the order used in build_nber_prior (i.e. VIX in column
        `self.vix_col`).

        If `transmat_prior` is omitted, uses a flat Dirichlet(1, 1, 1)
        prior (i.e. no shrinkage). To get the NBER-informed prior, call
        `build_nber_prior` and pass the result here.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (T, n_features); got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf; clean upstream before fit")

        # hmmlearn expects scalar OR (K,) / (K, K) priors
        if transmat_prior is None:
            transmat_prior_arg = 1.0
        else:
            transmat_prior = np.asarray(transmat_prior, dtype=np.float64)
            if transmat_prior.shape != (self.n_states, self.n_states):
                raise ValueError(
                    f"transmat_prior must be ({self.n_states}, {self.n_states}); "
                    f"got {transmat_prior.shape}"
                )
            if (transmat_prior <= 0).any():
                raise ValueError("transmat_prior entries must all be > 0 (Dirichlet)")
            transmat_prior_arg = transmat_prior

        if startprob_prior is None:
            startprob_prior_arg = 1.0
        else:
            startprob_prior_arg = np.asarray(startprob_prior, dtype=np.float64)

        m = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            transmat_prior=transmat_prior_arg,
            startprob_prior=startprob_prior_arg,
            init_params="stmc",   # randomize startprob/transmat/means/covars
            params="stmc",        # learn all four
            implementation="log",
        )
        m.fit(X)
        # Reorder so state 0 = lowest VIX (bull), state 2 = highest VIX (crisis)
        _relabel_by_feature(m, feature_index=self.vix_col, ascending=True)
        self._model = m
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior probabilities P(s_t | x_{1:T}) from forward-backward smoothing.
        Returns shape (T, n_states) where columns are in bull/bear/crisis order.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64).reshape(-1, self._model.means_.shape[1])
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most-likely state sequence via Viterbi. Returns (T,) int array."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64).reshape(-1, self._model.means_.shape[1])
        return self._model.predict(X)

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood of the observation sequence under the fitted model."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64).reshape(-1, self._model.means_.shape[1])
        return float(self._model.score(X))

    # ── inspection ──

    def summary(self) -> RegimeSummary:
        """Produce a human-readable summary used by the calibration report."""
        self._check_fitted()
        # Stationary distribution: solve π = π @ T
        eigvals, eigvecs = np.linalg.eig(self._model.transmat_.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stationary = np.real(eigvecs[:, idx])
        stationary = stationary / stationary.sum()
        # Mean dwell per state (geometric distribution on diagonal)
        diag = np.diag(self._model.transmat_)
        dwell = 1.0 / np.maximum(1.0 - diag, 1e-12)
        return RegimeSummary(
            state_labels=list(DEFAULT_STATE_LABELS),
            stationary_distribution=stationary.tolist(),
            mean_dwell_months=dwell.tolist(),
            state_means=self._model.means_.tolist(),
            transmat=self._model.transmat_.tolist(),
            n_iter=int(self._model.monitor_.iter),
            converged=bool(self._model.monitor_.converged),
            feature_names=self.feature_names or [],
        )

    # ── persistence ──

    def save(self, path: str | Path) -> None:
        """Pickle the fitted model + metadata. Companion JSON summary alongside."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "n_states": self.n_states,
            "vix_col": self.vix_col,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, path)
        # Companion JSON summary for human inspection
        summary_path = path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(self.summary().to_dict(), indent=2, default=float))

    @classmethod
    def load(cls, path: str | Path) -> "RegimeHMM":
        """Load a previously-saved RegimeHMM."""
        payload = joblib.load(Path(path))
        inst = cls(
            n_states=payload["n_states"],
            vix_col=payload["vix_col"],
            covariance_type=payload["covariance_type"],
            n_iter=payload["n_iter"],
            tol=payload["tol"],
            random_state=payload["random_state"],
            feature_names=payload["feature_names"],
        )
        inst._model = payload["model"]
        inst._fitted = True
        return inst

    # ── internals ──

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("RegimeHMM has not been fitted; call .fit(X) first")


__all__ = [
    "RegimeHMM",
    "RegimeSummary",
    "build_nber_prior",
    "DEFAULT_STATE_LABELS",
    "NBER_MEDIAN_EXPANSION_MONTHS",
    "NBER_MEDIAN_RECESSION_MONTHS",
]
