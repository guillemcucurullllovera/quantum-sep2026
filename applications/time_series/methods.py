"""Per-method feature-extraction adapters used by the unified dispatcher.

Every quantum or classical method exposes the same shape: an unsupervised
sequence -> feature-matrix map, computed once over the whole series
(causal by construction -- feature[t] only ever depends on X[0..t]), plus a
shared ridge-regression readout that is the only thing ever refit per
walk-forward fold. This mirrors the compute-once-refit-per-fold discipline
already used by algorithms.qrc.training.walk_forward_evaluate and
classical.reservoir.esn.EchoStateNetwork (see that module's docstring).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from algorithms.qrc.encoders import encode_sequence
from algorithms.qrc.gen2_v1_original import qrc_v1 as gen2_original
from algorithms.qrc.reservoir import (
    QuantumReservoir,
    QuantumReservoirConfig,
    evolution_operator_stable,
    qrc_sequence_features,
)
from algorithms.qrc import reservoir_math as gen1_arxiv
from algorithms.qrc.reservoir_math import ridge_fit as _arxiv_ridge_fit
from algorithms.qrc.reservoir_math import ridge_predict as _arxiv_ridge_predict
from applications.time_series.domain import MethodChoice, TimeSeriesForecastSpec
from applications.time_series.errors import UnknownForecastMethodError
from classical.reservoir.esn import ESNConfig, EchoStateNetwork
from classical.time_series.mamba_ssm_baseline import _SelectiveSSM
from core.errors import ValidationError


@dataclass
class ExtractedFeatures:
    features: np.ndarray  # (T', d) unsupervised feature matrix
    y_aligned: np.ndarray  # (T',) target aligned to features rows
    notes: str | None = None


def _extract_qrc_gen2(X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec) -> ExtractedFeatures:
    """Canonical current implementation: persistent-state architecture,
    spectral-radius-normalized couplings (the Gen2 bug fix)."""
    cfg = QuantumReservoirConfig(
        n1=spec.qrc.n1, n2=spec.qrc.n2, alpha=spec.qrc.alpha, tau=spec.qrc.tau,
        field_nu=spec.qrc.field_nu, seed=spec.seed,
    )
    angles = encode_sequence(X, n1=cfg.n1)
    reservoir = QuantumReservoir(cfg).build()
    feats = reservoir.sequence_features(angles, washout=0)
    return ExtractedFeatures(feats, y[: len(feats)], notes="gen2 persistent-state, spectral-radius-normalized couplings")


def _extract_qrc_gen2_original(X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec) -> ExtractedFeatures:
    """Preserved pre-refactor implementation (algorithms/qrc/gen2_v1_original)
    -- same math as QRC_GEN2, different code path; a parity check on the
    Gen2 refactor rather than an architecturally distinct method."""
    cfg = spec.qrc
    n = cfg.n1 + cfg.n2
    rng = np.random.default_rng(spec.seed)
    J = gen2_original.random_couplings(n, rng, cfg.alpha)
    H = gen2_original.build_ising_hamiltonian(n, J, cfg.field_nu)
    U = gen2_original.evolution_operator(H, cfg.tau)
    U_dag = U.conj().T
    z_diags = gen2_original.precompute_z_diagonals(n)
    angles = encode_sequence(X, n1=cfg.n1)
    feats = gen2_original.qrc_sequence_features(angles, U, cfg.n1, cfg.n2, washout=0, U_dag=U_dag, z_diags=z_diags)
    return ExtractedFeatures(feats, y[: len(feats)], notes="gen2 pre-refactor implementation (parity check against gen2)")


def _extract_qrc_gen1_arxiv(X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec) -> ExtractedFeatures:
    """Paper-faithful replication architecture: windowed (not
    persistent-state) forward pass with UNnormalized couplings, matching
    algorithms_extended/qrc_arxiv_replication exactly. That component's own
    THIRD_PARTY_NOTICES.md documents it failed its own accuracy gate (best MSE 0.2066
    vs paper target 0.105); its result here is reported as-is, not hidden or
    softened by the dispatcher.
    """
    cfg = spec.qrc
    n = cfg.n1 + cfg.n2
    window = cfg.arxiv_window
    rng = np.random.default_rng(spec.seed)
    J = gen1_arxiv.random_couplings(n, rng)  # NOTE: no spectral-radius normalization -- see THIRD_PARTY_NOTICES.md
    H = gen1_arxiv.build_ising_hamiltonian(n, J, field_nu=cfg.field_nu)
    U = gen1_arxiv.evolution_operator(H, tau=cfg.tau)
    U_dag = U.conj().T
    z_diags = gen1_arxiv.precompute_z_diagonals(n)
    angles = encode_sequence(X, n1=cfg.n1)
    T = len(angles)
    if T <= window:
        raise ValidationError(f"Series too short ({T} rows) for arxiv_window={window}")
    feats = np.empty((T - window, n), dtype=np.float64)
    for i, t in enumerate(range(window, T)):
        feats[i] = gen1_arxiv.qrc_forward(angles[t - window : t], U, cfg.n1, cfg.n2, U_dag, z_diags)
    return ExtractedFeatures(
        feats, y[window:],
        notes="paper-faithful windowed architecture, unnormalized couplings (documented gate failure, see THIRD_PARTY_NOTICES.md)",
    )


def _extract_qrc_gen1_informed(X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec) -> ExtractedFeatures:
    """Ablation this platform adds: Gen2's persistent-state architecture with
    Gen1's UNnormalized coupling-matrix generation. This isolates and tests
    the open question `algorithms/qrc/reservoir.py`'s own docstring raises
    but leaves untested -- whether the spectral-radius normalization fix
    (found during Gen2 development) matters independently of the
    windowed-vs-persistent-state architecture change.
    """
    cfg = spec.qrc
    n = cfg.n1 + cfg.n2
    rng = np.random.default_rng(spec.seed)
    J = gen1_arxiv.random_couplings(n, rng)  # unnormalized, Gen1-style
    H = gen1_arxiv.build_ising_hamiltonian(n, J, field_nu=cfg.field_nu)
    U = evolution_operator_stable(H, cfg.tau)  # Gen2's eigh-based evolution (same math as gen1_arxiv's expm, faster)
    angles = encode_sequence(X, n1=cfg.n1)
    feats = qrc_sequence_features(angles, U, cfg.n1, cfg.n2, washout=0)
    return ExtractedFeatures(
        feats, y[: len(feats)],
        notes="ablation: persistent-state architecture (gen2) + unnormalized couplings (gen1) -- tests reservoir.py's open question",
    )


def _extract_esn(X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec) -> ExtractedFeatures:
    cfg = ESNConfig(
        reservoir_size=spec.esn.reservoir_size, spectral_radius=spec.esn.spectral_radius,
        leak_rate=spec.esn.leak_rate, input_scaling=spec.esn.input_scaling,
        density=spec.esn.density, seed=spec.seed,
    )
    esn = EchoStateNetwork(cfg).build(n_inputs=X.shape[1])
    states = esn.compute_states(X)
    return ExtractedFeatures(states, y, notes="classical echo state network baseline")


def _extract_mamba(X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec) -> ExtractedFeatures:
    """Adapts classical.time_series.mamba_ssm_baseline's selective-SSM layer
    (built there for a 3-way classification head) into a regression feature
    extractor: the SSM's fixed random selective dynamics (log_A, W_B, W_C,
    W_dt -- untrained, exactly as ESN's reservoir weights are untrained) map
    the input sequence to a (T, d_model) feature trajectory, and this
    module's shared ridge readout (identical to every other method here)
    replaces the classification head. Reaches into that module's
    module-private `_SelectiveSSM` deliberately, documented here rather than
    silently: no public regression-oriented Mamba feature extractor exists
    upstream, and this is not a claim of reproducing Gu & Dao's trained,
    backprop-through-the-recurrence Mamba -- see that module's docstring.
    """
    rng = np.random.default_rng(spec.seed)
    d_model = X.shape[1]
    layers = [_SelectiveSSM(d_model, spec.mamba.d_state, spec.mamba.expand, rng) for _ in range(spec.mamba.n_layers)]
    feats = np.asarray(X, dtype=float)
    for layer in layers:
        feats = layer.forward(feats)
    return ExtractedFeatures(feats, y, notes="fixed-random selective-SSM features (Mamba dynamics, untrained) + ridge readout")


_EXTRACTORS: dict[MethodChoice, Callable[[np.ndarray, np.ndarray, TimeSeriesForecastSpec], ExtractedFeatures]] = {
    MethodChoice.QRC_GEN2: _extract_qrc_gen2,
    MethodChoice.QRC_GEN2_ORIGINAL: _extract_qrc_gen2_original,
    MethodChoice.QRC_GEN1_ARXIV: _extract_qrc_gen1_arxiv,
    MethodChoice.QRC_GEN1_INFORMED: _extract_qrc_gen1_informed,
    MethodChoice.ESN: _extract_esn,
    MethodChoice.MAMBA_SSM: _extract_mamba,
}


def extract_features(method: MethodChoice, X: np.ndarray, y: np.ndarray, spec: TimeSeriesForecastSpec) -> ExtractedFeatures:
    try:
        fn = _EXTRACTORS[method]
    except KeyError as e:
        raise UnknownForecastMethodError(
            f"No feature extractor registered for {method!r}. Registered: {sorted(m.value for m in _EXTRACTORS)}"
        ) from e
    return fn(X, y, spec)


def ridge_fit(M: np.ndarray, y: np.ndarray, delta: float) -> np.ndarray:
    """Shared ridge-regression readout: M is (d_features, n_samples), with a
    bias row appended -- same convention as
    algorithms_extended.qrc_arxiv_replication.reservoir_core.ridge_fit, so
    every method here is scored through the identical readout."""
    return _arxiv_ridge_fit(M, y, delta=delta, bias=True)


def ridge_predict(W: np.ndarray, M: np.ndarray) -> np.ndarray:
    return _arxiv_ridge_predict(W, M, bias=True)


__all__ = ["ExtractedFeatures", "extract_features", "ridge_fit", "ridge_predict"]
