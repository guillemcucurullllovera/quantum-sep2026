"""Track C harness: fair QRC-vs-ESN comparison on the synthetic realized-vol
proxy task (`EXP-C1`).

**Read this before changing any default here.** `algorithms/qrc/reservoir.py`
and `classical/reservoir/esn.py` are reused verbatim (not modified) per the
QUANTUM-1B plan -- this module only *wraps* them.

## Readout-dimension matching methodology (read `QuantumReservoir` before
assuming anything about its feature count)

`QuantumReservoir.windowed_features` / `.sequence_features` return, per
timestep, a vector of length `config.n = config.n1 + config.n2` -- the
Z-expectation values `z_diags @ diag(rho)` for every qubit (see
`algorithms/qrc/reservoir_math.py::qrc_forward` /
`algorithms/qrc/reservoir.py::qrc_sequence_features`). **This is NOT
`4**n2`.** The hidden state `rho_hidden` carried between steps is
internally a `4**n2`-real-number object (the existing Q10 audit,
`docs/research/Q10_AUDIT__classical-simulability.md` line ~50, makes this
exact point: "the only object surviving each timestep is `rho_hidden` ...
= 4^n2 real numbers ... [but] readout touches only diagonal populations"),
but the *public* readout the class actually exposes to a ridge head is
only the `n1+n2` Z-expectation scalars. There is no public API on
`QuantumReservoir` to pull out the full `4**n2`-dimensional density-matrix
readout, and this module does not reach into the class's internals to add
one (the brief requires wrapping, not modifying, `reservoir.py`).

Consequently, **readout dimension D is matched here as D = n1 + n2
(total qubit count)**, not via any `4**n2` projection. This is a
deliberate, documented correction to the preregistration's informal
description ("QRC's n2 hidden qubits set so `4^n2 >= D` readout features
project down to D") -- that description does not match what
`QuantumReservoir`'s actual public interface produces, discovered while
reading the class (per the task's own instruction to verify, not assume).
It is logged as a preregistration clarification in
`docs/quantum/QUANTUM1B_QRC_ESN.md`, not silently substituted.

Given D = n1 + n2, this module fixes **n1 = 1** (a single scalar input
qubit encoding the current raw observation, matching the ESN's
single-channel `u_t` input convention -- both reservoirs see one scalar
per timestep and rely entirely on their own recurrent/hidden state for
memory, which is the fair, apples-to-apples reservoir-computing setup)
and sets **n2 = D - 1** to hit the target total. Total system dimension
for the *full* density-matrix simulation is `2**(n1+n2) = 2**D`, so this
module also caps which D values are computationally reachable -- see
`DEFAULT_READOUT_DIMS` below.

## Preregistered readout-dimension grid: DEVIATION

`EXP-C1` (see `docs/quantum/QUANTUM1B_PREREGISTRATION.md` and
`experiments/quantum1b/registry/EXP-C1.json`) specifies readout dims
`D in {4, 8, 16}`. **D=16 is dropped here; D=6 is added.** This is a hard
compute/memory infeasibility, not a "slow but doable, trimmed for
convenience" judgment call: `QuantumReservoir` represents its reservoir
state as a *full density matrix* of the `n1+n2`-qubit system (see
`qrc_sequence_features`: `rho = np.kron(rho_input, rho_hidden)`, evolved
by `U @ rho @ U_dag` every step). At D=16 that is a `65536 x 65536`
complex128 matrix -- roughly 68.7 GB just to hold `rho` once, before
accounting for `U`/`H` (same size) or the O(dim^3) cost of `U @ rho @
U_dag` at every one of ~2000 timesteps. This is not reachable on any
machine available to this task, let alone inside the preregistered
<10-minute compute budget. Empirical timing on this machine (see
`docs/quantum/QUANTUM1B_QRC_ESN.md` for the full table): D=8 (dim=256)
costs ~16ms/timestep; D=10 (dim=1024) already costs ~155ms/timestep
(~5 minutes for one seed's full 2000-step sequence alone). D=16 was never
attempted -- the `2**16`-dimensional density matrix does not fit in
memory. The grid used here, `DEFAULT_READOUT_DIMS = (4, 6, 8)`, keeps the
preregistered cardinality (3 points) and keeps D=4 and D=8 from the
original grid, inserting D=6 between them rather than reaching for D=16.
Logged per `QUANTUM1B_PREREGISTRATION.md` cross-cutting rule 5
(deviations discovered during implementation, before any result exists,
are documented -- not silently absorbed).

## Compute-budget design choices (all documented, none silent)

- `DEFAULT_SERIES_LEN = 2000` (meets EXP-C1's ">= 2000 points" implicit
  requirement per the task brief).
- Memory-capacity/IPC probe sequences use a *shorter* i.i.d. input
  (`MC_INPUT_LEN = 300`, not the full 2000-point task series) -- this is
  a standard Dambre-2012-style diagnostic run, not the forecasting task
  itself, and keeping it short is what keeps the full EXP-C1 grid inside
  the <10 minute budget once D=8 is included.
- A fixed washout (`WASHOUT = 50` steps, truncated further for short
  splits) is applied at the start of every *training* fit for the
  persistent-state reservoirs (ESN, QRC) to let recurrent state settle
  from its zero initial condition -- never applied at the train/test
  boundary itself, since reservoir state continuity across that boundary
  is exactly what "expanding-window, no re-warm-up" is supposed to test.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from algorithms.qrc.encoders import encode_sequence
from algorithms.qrc.reservoir import QuantumReservoir, QuantumReservoirConfig
from algorithms.qrc.reservoir_math import ridge_fit, ridge_predict
from classical.reservoir.esn import ESNConfig, EchoStateNetwork
from experiments.quantum1b.synthetic_vol_task import generate_synthetic_realized_vol_proxy

# -- Preregistered / documented-deviation constants -----------------------

DEFAULT_READOUT_DIMS: tuple[int, ...] = (4, 6, 8)  # preregistered (4, 8, 16) -- see module docstring
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)  # preregistered, unchanged
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)  # preregistered, unchanged
DEFAULT_SERIES_LEN: int = 2000
DEFAULT_TRAIN_FRAC: float = 0.7  # preregistered 70/30 split

N_LAGS_RIDGE_RAW: int = 10  # lag window for the ridge/linear-on-raw-features baseline
WASHOUT: int = 50  # reservoir warm-up steps excluded from training fits
VAL_FRAC: int = 0.2  # fraction of the TRAIN portion held out for alpha selection

MC_INPUT_LEN: int = 300  # i.i.d. probe length for the memory-capacity/IPC diagnostic
MC_MAX_DELAY_MULT: int = 3  # max delay tested = MC_MAX_DELAY_MULT * readout_dim
MC_WASHOUT: int = 20  # reservoir settle time before MC regression targets start


# -- Result container -------------------------------------------------------


@dataclass
class ForecastResult:
    method: str  # "persistence" | "ridge_raw" | "esn" | "qrc"
    oos_mse: float
    qlike: float | None
    readout_dim: int
    training_runtime_s: float
    seed: int
    ridge_alpha: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "oos_mse": self.oos_mse,
            "qlike": self.qlike,
            "readout_dim": self.readout_dim,
            "training_runtime_s": self.training_runtime_s,
            "seed": self.seed,
            "ridge_alpha": self.ridge_alpha,
            "extra": _json_safe(self.extra),
        }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# -- Metrics ------------------------------------------------------------


def qlike_loss(actual: np.ndarray, pred: np.ndarray) -> float | None:
    """Patton (2011) QLIKE loss, averaged over points where `pred > 0`:

        qlike_t = actual_t/pred_t - log(actual_t/pred_t) - 1

    Meaningful ONLY for non-negative, variance/volatility-like targets
    (our synthetic proxy is always > 0 by construction --
    `synthetic_vol_task.FLOOR_EPS`). Returns `None` (rather than raising)
    when no point has `pred > 0`, since QLIKE here is a secondary
    diagnostic layered on top of MSE, not a blocking requirement.
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = pred > 0
    if not np.any(mask):
        return None
    ratio = np.clip(actual[mask] / pred[mask], 1e-12, None)
    vals = ratio - np.log(ratio) - 1.0
    return float(np.mean(vals))


def _mc_capacity_for_delay(X: np.ndarray, target: np.ndarray, train_frac: float = 0.7, delta: float = 1e-6) -> float:
    """Out-of-sample R^2 (clipped to [0,1]) of a linear ridge reconstruction
    of `target` from reservoir-state features `X`, Dambre (2012) style."""
    m = len(target)
    if m < 10:
        return 0.0
    split = int(m * train_frac)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = target[:split], target[split:]
    if len(y_te) < 3:
        return 0.0
    W = ridge_fit(X_tr.T, y_tr, delta=delta, bias=True)
    pred = ridge_predict(W, X_te.T, bias=True)
    sse = float(np.sum((y_te - pred) ** 2))
    sst = float(np.sum((y_te - y_te.mean()) ** 2))
    if sst < 1e-12:
        return 0.0
    r2 = 1.0 - sse / sst
    return float(np.clip(r2, 0.0, 1.0))


def memory_capacity(states: np.ndarray, u: np.ndarray, max_delay: int, washout: int = MC_WASHOUT) -> dict[str, Any]:
    """Dambre (2012)-style memory-capacity / IPC diagnostic: for delays
    k = 0..max_delay, linearly reconstruct u_{t-k} from reservoir state
    x_t (out-of-sample R^2, clipped to [0,1] per delay), report the
    per-delay curve and `total = sum_k MC_k`.
    """
    L = len(u)
    per_delay = []
    for k in range(0, max_delay + 1):
        t_start = max(washout, k)
        idx_t = np.arange(t_start, L)
        if len(idx_t) < 10:
            per_delay.append({"delay": k, "capacity": 0.0})
            continue
        X = states[idx_t]
        target = u[idx_t - k]
        cap = _mc_capacity_for_delay(X, target)
        per_delay.append({"delay": k, "capacity": cap})
    total = float(sum(d["capacity"] for d in per_delay))
    return {"total": total, "per_delay": per_delay, "max_delay": max_delay}


# -- Shared helpers -----------------------------------------------------


def _expanding_split(L: int, train_frac: float = DEFAULT_TRAIN_FRAC) -> tuple[np.ndarray, np.ndarray]:
    split = int(round(train_frac * L))
    split = max(1, min(L - 1, split))
    return np.arange(0, split), np.arange(split, L)


def _standardize(x: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
    mu = float(x[fit_idx].mean())
    sigma = float(x[fit_idx].std()) + 1e-12
    return (x - mu) / sigma


def _select_alpha_generic(
    fit_fn: Callable[[float], None],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    inner_train_idx: np.ndarray,
    val_idx: np.ndarray,
    y: np.ndarray,
    alpha_grid: tuple[float, ...],
) -> tuple[float, float]:
    """Generic small-grid alpha selection: fit on `inner_train_idx`,
    evaluate MSE on `val_idx`, return (best_alpha, best_val_mse)."""
    best_alpha, best_val_mse = alpha_grid[0], float("inf")
    for a in alpha_grid:
        fit_fn(a)
        pred_val = predict_fn(val_idx)
        vmse = float(np.mean((y[val_idx] - pred_val) ** 2))
        if vmse < best_val_mse:
            best_val_mse, best_alpha = vmse, a
    return best_alpha, best_val_mse


# -- Baselines ------------------------------------------------------------


def run_persistence_baseline(series: np.ndarray, split: int, seed: int = 0) -> ForecastResult:
    """Naive last-value forecast: y_hat_t = y_{t-1} (i.e. predict
    `series[t]` with `series[t-1]`)."""
    t0 = time.time()
    y = series[1:]
    pred_all = series[:-1]
    L = len(y)
    test_idx = np.arange(split, L)
    actual, pred = y[test_idx], pred_all[test_idx]
    mse = float(np.mean((actual - pred) ** 2))
    ql = qlike_loss(actual, pred)
    return ForecastResult(
        method="persistence", oos_mse=mse, qlike=ql, readout_dim=0,
        training_runtime_s=time.time() - t0, seed=seed, ridge_alpha=None, extra={},
    )


def run_ridge_raw_baseline(
    series: np.ndarray, split: int, alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    n_lags: int = N_LAGS_RIDGE_RAW, seed: int = 0, val_frac: float = VAL_FRAC,
) -> ForecastResult:
    """Ridge regression on an `n_lags`-length raw-value lag window
    (standardized using train-only statistics), alpha selected by a
    simple held-out train/val split within the training portion."""
    t0 = time.time()
    y = series[1:]
    L = len(y)
    valid_t = np.arange(n_lags - 1, L)  # rows with enough history
    lag_feats = np.stack([series[t - n_lags + 1 : t + 1] for t in valid_t])  # (m, n_lags)

    train_mask = valid_t < split
    train_rows = np.where(train_mask)[0]
    test_rows = np.where(~train_mask)[0]

    mu = lag_feats[train_rows].mean(axis=0)
    sigma = lag_feats[train_rows].std(axis=0) + 1e-12
    feats_std = (lag_feats - mu) / sigma
    targets = y[valid_t]

    val_len = max(1, int(val_frac * len(train_rows)))
    inner_train_rows = train_rows[: len(train_rows) - val_len]
    val_rows = train_rows[len(train_rows) - val_len :]

    state = {"W": None}

    def fit_fn(a: float) -> None:
        state["W"] = ridge_fit(feats_std[inner_train_rows].T, targets[inner_train_rows], delta=a, bias=True)

    def predict_fn(idx: np.ndarray) -> np.ndarray:
        return ridge_predict(state["W"], feats_std[idx].T, bias=True)

    best_alpha, best_val_mse = _select_alpha_generic(fit_fn, predict_fn, inner_train_rows, val_rows, targets, alpha_grid)

    W = ridge_fit(feats_std[train_rows].T, targets[train_rows], delta=best_alpha, bias=True)
    pred_test = ridge_predict(W, feats_std[test_rows].T, bias=True)
    actual_test = targets[test_rows]
    mse = float(np.mean((actual_test - pred_test) ** 2))
    ql = qlike_loss(actual_test, pred_test)

    return ForecastResult(
        method="ridge_raw", oos_mse=mse, qlike=ql, readout_dim=n_lags,
        training_runtime_s=time.time() - t0, seed=seed, ridge_alpha=best_alpha,
        extra={"val_mse": best_val_mse, "n_lags": n_lags},
    )


# -- ESN --------------------------------------------------------------------


def run_esn(
    series: np.ndarray, split: int, readout_dim: int, alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    seed: int = 0, val_frac: float = VAL_FRAC,
) -> ForecastResult:
    """Classical ESN (`classical/reservoir/esn.py`), reservoir/readout size
    = `readout_dim`, ridge alpha selected from `alpha_grid` via a
    train/val split within the training portion."""
    t0 = time.time()
    y = series[1:]
    L = len(y)
    U = series[:-1].reshape(-1, 1)
    train_idx, test_idx = np.arange(0, split), np.arange(split, L)
    U_std = _standardize(U[:, 0], train_idx).reshape(-1, 1)

    # very small reservoirs need higher connection density than the
    # ESNConfig default (0.1) or the reservoir matrix can end up all-zero
    # (ValidationError: near-zero spectral radius) -- documented adjustment.
    density = float(min(1.0, max(0.3, 4.0 / readout_dim)))
    cfg = ESNConfig(
        reservoir_size=readout_dim, spectral_radius=0.9, leak_rate=0.3,
        input_scaling=1.0, ridge_lambda=alpha_grid[0], density=density, seed=seed,
    )
    esn = EchoStateNetwork(cfg).build(n_inputs=1)
    states = esn.compute_states(U_std)

    val_len = max(1, int(val_frac * len(train_idx)))
    inner_train_idx = train_idx[: len(train_idx) - val_len]
    val_idx = train_idx[len(train_idx) - val_len :]
    wo_inner = min(WASHOUT, max(0, len(inner_train_idx) // 4))

    def fit_fn(a: float) -> None:
        esn.config.ridge_lambda = a
        esn.fit_readout(states[inner_train_idx], y[inner_train_idx], washout=wo_inner)

    def predict_fn(idx: np.ndarray) -> np.ndarray:
        return esn.predict(states[idx])

    best_alpha, best_val_mse = _select_alpha_generic(fit_fn, predict_fn, inner_train_idx, val_idx, y, alpha_grid)

    wo_full = min(WASHOUT, max(0, len(train_idx) // 4))
    esn.config.ridge_lambda = best_alpha
    esn.fit_readout(states[train_idx], y[train_idx], washout=wo_full)
    pred_test = esn.predict(states[test_idx])
    mse = float(np.mean((y[test_idx] - pred_test) ** 2))
    ql = qlike_loss(y[test_idx], pred_test)

    probe = np.random.default_rng(seed * 1000 + 7).uniform(-1.0, 1.0, size=MC_INPUT_LEN).reshape(-1, 1)
    probe_states = esn.compute_states(probe)
    mc = memory_capacity(probe_states, probe.ravel(), max_delay=MC_MAX_DELAY_MULT * readout_dim)

    return ForecastResult(
        method="esn", oos_mse=mse, qlike=ql, readout_dim=readout_dim,
        training_runtime_s=time.time() - t0, seed=seed, ridge_alpha=best_alpha,
        extra={"memory_capacity": mc, "val_mse": best_val_mse, "density": density},
    )


# -- QRC ----------------------------------------------------------------


def run_qrc(
    series: np.ndarray, split: int, readout_dim: int, alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    seed: int = 0, val_frac: float = VAL_FRAC,
) -> ForecastResult:
    """`QuantumReservoir` (`algorithms/qrc/reservoir.py`), readout
    dimension = `readout_dim` via `n1=1` (scalar input) + `n2=readout_dim-1`
    hidden qubits (so `n1+n2 == readout_dim` Z-expectation features --
    see module docstring for why this is the correct readout-matching
    rule for THIS class's actual public API, not a `4**n2` projection).
    """
    t0 = time.time()
    y = series[1:]
    L = len(y)
    U = series[:-1]
    train_idx, test_idx = (np.arange(0, split), np.arange(split, L))
    U_std = _standardize(U, train_idx)

    n1 = 1
    n2 = max(1, readout_dim - n1)
    cfg = QuantumReservoirConfig(n1=n1, n2=n2, seed=seed)
    res = QuantumReservoir(cfg).build()
    angles = encode_sequence(U_std.reshape(-1, 1), n1=n1)
    features = res.sequence_features(angles, washout=0)  # (L, n1+n2) == (L, readout_dim)

    val_len = max(1, int(val_frac * len(train_idx)))
    inner_train_idx = train_idx[: len(train_idx) - val_len]
    val_idx = train_idx[len(train_idx) - val_len :]
    wo_inner = min(WASHOUT, max(0, len(inner_train_idx) // 4))

    state = {"W": None}

    def fit_fn(a: float) -> None:
        tr = inner_train_idx[wo_inner:] if wo_inner < len(inner_train_idx) else inner_train_idx
        state["W"] = ridge_fit(features[tr].T, y[tr], delta=a, bias=True)

    def predict_fn(idx: np.ndarray) -> np.ndarray:
        return ridge_predict(state["W"], features[idx].T, bias=True)

    best_alpha, best_val_mse = _select_alpha_generic(fit_fn, predict_fn, inner_train_idx, val_idx, y, alpha_grid)

    wo_full = min(WASHOUT, max(0, len(train_idx) // 4))
    tr_full = train_idx[wo_full:] if wo_full < len(train_idx) else train_idx
    W = ridge_fit(features[tr_full].T, y[tr_full], delta=best_alpha, bias=True)
    pred_test = ridge_predict(W, features[test_idx].T, bias=True)
    mse = float(np.mean((y[test_idx] - pred_test) ** 2))
    ql = qlike_loss(y[test_idx], pred_test)

    probe = np.random.default_rng(seed * 1000 + 7).uniform(-1.0, 1.0, size=MC_INPUT_LEN)
    probe_angles = encode_sequence(probe.reshape(-1, 1), n1=n1)
    probe_states = res.sequence_features(probe_angles, washout=0)
    mc = memory_capacity(probe_states, probe, max_delay=MC_MAX_DELAY_MULT * readout_dim)

    return ForecastResult(
        method="qrc", oos_mse=mse, qlike=ql, readout_dim=readout_dim,
        training_runtime_s=time.time() - t0, seed=seed, ridge_alpha=best_alpha,
        extra={"memory_capacity": mc, "val_mse": best_val_mse, "n1": n1, "n2": n2},
    )


# -- Full comparison grid -----------------------------------------------


def compare_qrc_vs_esn(
    readout_dims: tuple[int, ...] = DEFAULT_READOUT_DIMS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    n_points: int = DEFAULT_SERIES_LEN,
    train_frac: float = DEFAULT_TRAIN_FRAC,
) -> dict[str, Any]:
    """Run persistence, ridge_raw, ESN, and QRC across the readout-dim x
    seed grid on the synthetic task. Returns per-seed `ForecastResult`
    lists (never aggregated away) plus paired ESN-vs-QRC MSE differences
    per readout dim, for downstream distributional/significance reporting
    -- "report distributions, never a single best-seed number" per the
    QUANTUM-1B brief.
    """
    persistence_results: list[ForecastResult] = []
    ridge_results: list[ForecastResult] = []
    esn_results: dict[int, list[ForecastResult]] = {D: [] for D in readout_dims}
    qrc_results: dict[int, list[ForecastResult]] = {D: [] for D in readout_dims}
    paired_diffs: dict[int, list[float]] = {D: [] for D in readout_dims}

    for seed in seeds:
        sv = generate_synthetic_realized_vol_proxy(n=n_points, seed=seed)
        series = sv.proxy
        L = len(series) - 1
        train_idx, _test_idx = _expanding_split(L, train_frac)
        split = len(train_idx)

        persistence_results.append(run_persistence_baseline(series, split, seed=seed))
        ridge_results.append(run_ridge_raw_baseline(series, split, alpha_grid=alpha_grid, seed=seed))

        for D in readout_dims:
            r_esn = run_esn(series, split, readout_dim=D, alpha_grid=alpha_grid, seed=seed)
            r_qrc = run_qrc(series, split, readout_dim=D, alpha_grid=alpha_grid, seed=seed)
            esn_results[D].append(r_esn)
            qrc_results[D].append(r_qrc)
            paired_diffs[D].append(r_esn.oos_mse - r_qrc.oos_mse)  # negative => ESN better (lower MSE)

    return {
        "persistence": persistence_results,
        "ridge_raw": ridge_results,
        "esn": esn_results,
        "qrc": qrc_results,
        "paired_esn_minus_qrc_mse": paired_diffs,
        "readout_dims": list(readout_dims),
        "seeds": list(seeds),
        "alpha_grid": list(alpha_grid),
        "n_points": n_points,
        "train_frac": train_frac,
    }


def serialize_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    def _ser_list(lst: list[ForecastResult]) -> list[dict[str, Any]]:
        return [r.to_json_dict() for r in lst]

    def _ser_by_D(d: dict[int, list[ForecastResult]]) -> dict[str, Any]:
        return {str(D): _ser_list(lst) for D, lst in d.items()}

    return {
        "persistence": _ser_list(comparison["persistence"]),
        "ridge_raw": _ser_list(comparison["ridge_raw"]),
        "esn": _ser_by_D(comparison["esn"]),
        "qrc": _ser_by_D(comparison["qrc"]),
        "paired_esn_minus_qrc_mse": {str(D): v for D, v in comparison["paired_esn_minus_qrc_mse"].items()},
        "readout_dims": comparison["readout_dims"],
        "seeds": comparison["seeds"],
        "alpha_grid": comparison["alpha_grid"],
        "n_points": comparison["n_points"],
        "train_frac": comparison["train_frac"],
    }


def paired_significance(diffs: list[float]) -> dict[str, Any]:
    """Paired comparison of ESN-vs-QRC out-of-sample MSE differences
    (ESN - QRC) across seeds, at one readout dimension.

    Primary test: two-sided one-sample t-test (`scipy.stats.ttest_1samp`)
    of the per-seed differences against 0 -- appropriate for EXP-C1's 5
    paired, continuous-valued seeds. A simple sign count (how many seeds
    favor each method) is also reported as a distribution-free robustness
    cross-check, never as the primary decision statistic -- consistent
    with "pick something appropriate and document it."
    """
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    n_esn_better = int(np.sum(arr < 0))
    n_qrc_better = int(np.sum(arr > 0))
    n_tied = int(np.sum(arr == 0))
    mean_diff = float(np.mean(arr)) if n else float("nan")
    std_diff = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    t_stat, p_value = float("nan"), float("nan")
    if n > 1 and std_diff > 0:
        from scipy import stats

        t_res = stats.ttest_1samp(arr, popmean=0.0)
        t_stat, p_value = float(t_res.statistic), float(t_res.pvalue)

    return {
        "n": n,
        "mean_diff_esn_minus_qrc_mse": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_esn_better": n_esn_better,
        "n_qrc_better": n_qrc_better,
        "n_tied": n_tied,
    }


# -- Real-financial-task execution: BLOCKED ------------------------------


class RealFinancialExecutionBlocked(RuntimeError):
    """Raised by `run_real_financial_task(...)`.

    Real financial-task execution of this harness is explicitly BLOCKED
    for QUANTUM-1B Track C.

    Reason: Model Lab's own data-readiness gate
    (`research-corpus/model-lab/work/readiness/READINESS_BRIEF.md`, D4
    ruling) records an operator-accepted HOLD on activating any
    questionable-license free-tier data source (Alpaca/Massive free tier
    not confirmed for commercial/internal-research rights; nothing
    purchased). There is currently no daily OHLCV for the AMC's six
    underlyings (SPY, QQQ, TLT, IEF, GLD, USO) held anywhere in the
    estate with confirmed research rights.
    `docs/quantum/QUANTUM1B_PREREGISTRATION.md` EXP-C1 anticipates exactly
    this fallback: build and validate the harness on a clearly-labeled
    synthetic task now, and block real-financial execution rather than
    fake it.

    To unblock: Model Lab must issue a RESEARCH_ONLY authority contract
    covering licensed daily OHLCV for SPY/QQQ/TLT/IEF/GLD/USO (see
    `docs/quantum/QUANTUM1B_PLAN.md` section 5 and
    `docs/quantum/QUANTUM1B_QRC_ESN.md`). Even once unblocked, running
    this harness on real data would still only reach evidence level B/C
    (simulator) with `financial_edge_claimed=False` unless a real,
    resolvable backtest artifact is produced per `core.evidence` -- that
    is a separate, higher bar this guard does not by itself satisfy.
    """


def run_real_financial_task(*args: Any, **kwargs: Any) -> None:
    """No-op guard: there is no real-data loader anywhere in this module
    to call in the first place. This function exists purely so the
    BLOCKED status is explicit and discoverable in code (importable,
    callable, raises a typed exception), not only asserted in prose or
    documentation. See `RealFinancialExecutionBlocked`'s docstring for
    the exact reason and what would unblock it.
    """
    raise RealFinancialExecutionBlocked(
        "Real financial-task execution is BLOCKED for QUANTUM-1B Track C: "
        "no licensed daily OHLCV for SPY/QQQ/TLT/IEF/GLD/USO is held with "
        "confirmed research rights (Model Lab readiness gate D4 ruling, "
        "research-corpus/model-lab/work/readiness/READINESS_BRIEF.md: HOLD "
        "on unconfirmed-license free-tier sources). Unblocking requires a "
        "Model Lab RESEARCH_ONLY authority contract for licensed OHLCV on "
        "these six underlyings. See QUANTUM1B_PREREGISTRATION.md EXP-C1 and "
        "docs/quantum/QUANTUM1B_QRC_ESN.md."
    )


__all__ = [
    "ForecastResult",
    "qlike_loss",
    "memory_capacity",
    "run_persistence_baseline",
    "run_ridge_raw_baseline",
    "run_esn",
    "run_qrc",
    "compare_qrc_vs_esn",
    "serialize_comparison",
    "paired_significance",
    "RealFinancialExecutionBlocked",
    "run_real_financial_task",
    "DEFAULT_READOUT_DIMS",
    "DEFAULT_SEEDS",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_SERIES_LEN",
    "DEFAULT_TRAIN_FRAC",
]
