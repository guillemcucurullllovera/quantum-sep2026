"""Regime-conditional hedging overlay: uses the VQC regime classifier's
predicted regime to decide a hedge sizing multiplier, and quantifies the
effect on tail risk via `applications.tail_risk.assess_tail_risk` (CVaR).

Genuine composition, not a stub: this is the first place in the platform
where a quantum classification OUTPUT feeds into a downstream portfolio
decision (hedge sizing) whose effect is then measured with the existing
quantum/classical tail-risk machinery -- closing the loop from "the model
predicts a regime" to "here is what that's worth."

Simplification, documented not hidden: hedge effectiveness is modeled as a
downside-truncation multiplier on simulated P&L (a protective put reduces
realized losses below its strike by roughly its notional-weighted payoff);
this is a standard first-order approximation, not a full options-pricing
simulation of the hedge itself (that's `applications.options`'s job, kept
separate to avoid entangling two different platform layers here).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.registry import APPLICATION_REGISTRY

from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from algorithms.vqc_qml.training import TrainingConfig, VQCTrainer
from applications.regime_detection.synthetic_regime_dataset import REGIMES
from applications.tail_risk.var_cvar import assess_tail_risk

# Hedge fraction applied to downside P&L per predicted regime: "trend_down"
# and "ranging" (chop/whipsaw) get a larger protective hedge than "trend_up".
DEFAULT_HEDGE_POLICY: dict[str, float] = {"trend_up": 0.0, "trend_down": 0.6, "ranging": 0.3}
HEDGE_EFFECTIVENESS = 0.7  # fraction of downside the hedge actually offsets, once applied


def _apply_hedge(pnl: np.ndarray, regime_labels: np.ndarray, policy: dict[str, float]) -> np.ndarray:
    hedged = pnl.copy()
    for i, regime_idx in enumerate(regime_labels):
        regime = REGIMES[int(regime_idx)]
        hedge_fraction = policy.get(regime, 0.0)
        if hedged[i] < 0:
            hedged[i] *= 1.0 - hedge_fraction * HEDGE_EFFECTIVENESS
    return hedged


@dataclass
class RegimeHedgeResult:
    fold: int
    regime_classifier_accuracy: float
    unhedged_cvar: float
    predicted_regime_hedge_cvar: float
    oracle_regime_hedge_cvar: float
    always_hedge_cvar: float
    predicted_hedge_captures_fraction_of_oracle_benefit: float


def run_regime_conditional_hedge(
    regime_dataset: FinancialDataset,
    pnl_samples: np.ndarray,
    n_qubits: int = 6,
    n_splits: int = 3,
    hedge_policy: dict[str, float] | None = None,
) -> list[RegimeHedgeResult]:
    """`pnl_samples` must have the same length as `regime_dataset` (one P&L
    observation per regime-labeled sample) -- the caller supplies both
    aligned, since this module does not itself generate P&L data.
    """
    policy = hedge_policy or DEFAULT_HEDGE_POLICY
    X, y = regime_dataset.features(), regime_dataset.targets()
    if len(pnl_samples) != len(y):
        from core.errors import ValidationError

        raise ValidationError(f"pnl_samples length {len(pnl_samples)} must match dataset length {len(y)}")
    splits = regime_dataset.split_walk_forward(n_splits=n_splits)
    ansatz = VQCAnsatz(n_qubits=n_qubits, version=VQCVersion.V5_REUPLOAD)

    results: list[RegimeHedgeResult] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        pnl_test = np.asarray(pnl_samples)[test_idx]

        trainer = VQCTrainer(ansatz, TrainingConfig()).fit(X_train, y_train)
        predicted_regimes = trainer.predict(X_test)
        clf_acc = trainer.accuracy(X_test, y_test)

        unhedged_cvar = assess_tail_risk(pnl_test, run_quantum=False, run_meta_agent_comparator=False).cvar
        predicted_hedge_pnl = _apply_hedge(pnl_test, predicted_regimes, policy)
        predicted_cvar = assess_tail_risk(predicted_hedge_pnl, run_quantum=False, run_meta_agent_comparator=False).cvar
        oracle_hedge_pnl = _apply_hedge(pnl_test, y_test, policy)
        oracle_cvar = assess_tail_risk(oracle_hedge_pnl, run_quantum=False, run_meta_agent_comparator=False).cvar
        always_hedge_pnl = pnl_test * np.where(pnl_test < 0, 1.0 - HEDGE_EFFECTIVENESS, 1.0)
        always_cvar = assess_tail_risk(always_hedge_pnl, run_quantum=False, run_meta_agent_comparator=False).cvar

        oracle_benefit = unhedged_cvar - oracle_cvar
        predicted_benefit = unhedged_cvar - predicted_cvar
        captured_fraction = float(predicted_benefit / oracle_benefit) if oracle_benefit > 1e-12 else float("nan")

        results.append(
            RegimeHedgeResult(
                fold=fold,
                regime_classifier_accuracy=clf_acc,
                unhedged_cvar=unhedged_cvar,
                predicted_regime_hedge_cvar=predicted_cvar,
                oracle_regime_hedge_cvar=oracle_cvar,
                always_hedge_cvar=always_cvar,
                predicted_hedge_captures_fraction_of_oracle_benefit=captured_fraction,
            )
        )
    return results


@APPLICATION_REGISTRY.register("regime_conditional_hedge")
class RegimeConditionalHedgeApp:
    """Registered application entry point (see core.registry / core.cli)."""

    @staticmethod
    def run(**kwargs) -> list[RegimeHedgeResult]:
        from applications.regime_detection.synthetic_regime_dataset import SyntheticRegimeDataset

        dataset = kwargs.pop("dataset", None) or SyntheticRegimeDataset()
        pnl_samples = kwargs.pop("pnl_samples", None)
        if pnl_samples is None:
            rng = np.random.default_rng(99)
            n = len(dataset.targets())
            regime_drift = {"trend_up": 0.02, "trend_down": -0.03, "ranging": 0.0}
            pnl_samples = np.array(
                [
                    rng.normal(regime_drift[REGIMES[int(r)]], 0.05)
                    for r in dataset.targets()
                ]
            )
        return run_regime_conditional_hedge(dataset, pnl_samples, **kwargs)
