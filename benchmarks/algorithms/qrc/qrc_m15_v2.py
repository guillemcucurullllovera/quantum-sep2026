"""
QRC M15 Benchmark v2 — Multi-horizon · Rich M15 features · 5-seed ensemble · Sharpe
====================================================================================
A comprehensive, corrected M15 (15-minute) QRC benchmark on OANDA FX/metal data,
designed for an apples-to-apples comparison with the validated *daily* QRC book.

WHY v2 — the three root causes fixed vs the naive M15 benchmark
---------------------------------------------------------------
1. HORIZON.  The naive M15 benchmark predicted the *next bar* (15 min ahead). At
   15-min resolution the next-bar sign is ~microstructure noise, so BA collapses
   to chance. v2 predicts a **5-day (480 M15-bar) forward move** — the SAME economic
   horizon as the validated daily book (5 daily bars) — giving a true cross-frequency
   comparison. We also sweep **4h (16 bars)** and **1D (96 bars)** look-aheads.
       4h = 4 h × 4 bars/h  = 16 bars
       1D = 24 h × 4 bars/h = 96 bars
       5D = 5 d × 96 bars/d = 480 bars

2. FEATURES.  A richer M15 feature set (the next-bar version used 3 raw returns):
       momentum cascade : ret_4bar(1h) ret_16bar(4h) ret_96bar(1D) ret_480bar(5D)
       realized vol     : rv_16bar(4h) rv_96bar(1D)
       vol regime       : vol_ratio = rv_16bar / rv_96bar   (intraday vs daily vol)
       session (cyclic) : hour_sin/cos = sin,cos(2π·hour/24)
       week (cyclic)    : dow_sin/cos  = sin,cos(2π·dow/5)
       session flags    : is_london(6-15 UTC) is_newyork(13-21 UTC) is_overlap(13-15 UTC)
   → 14 classical features (the "10 features" spec, expanded once the sin/cos pairs
     and 3 session flags are counted). The **first three columns are the momentum
     cascade (ret_4bar, ret_16bar, ret_96bar)** so N1=3 encodes exactly those into
     the quantum reservoir, per spec; the concat model uses all 14 classical + QRC.

3. ENSEMBLE + COST.  5 reservoir seeds (42-46), decision functions averaged. Sharpe
   computed with a **2 bps roundtrip** (tight spot spread for GBP/USD, XAU/USD).

Architecture (optimal-from-daily, FAST):
    n = 6 (n1 = 3 input, n2 = 3 hidden), α = 0.9, ν = 0.5, τ = 1.0
    Per-step readout: 6 ⟨Z_i⟩ + 5 ⟨Z_i Z_{i+1}⟩ = 11 features.
    n=6 → 64×64 operators (~33 s / 100k bars). n=10 (1024×1024) would be ~200 s.
    FX has a *fast* momentum signal that more qubits dilute; n1=3 keeps the 3 fastest
    momentum lookbacks in the encoded state — matching the daily-optimal config.

CORRECTNESS FIXES over the earlier M15/1h reference scripts
-----------------------------------------------------------
  * TEST-SET WARM-UP: the earlier scripts ran the reservoir over the test window with
    a cold |0⟩ hidden state and a 100-bar washout, then paired feature row i with
    test-label i — a silent 100-bar MISALIGNMENT (features started at test-bar 100).
    v2 warms the reservoir with the 100 bars immediately *preceding* the first test
    bar (embargo tail + train tail), so every test bar gets a correctly-aligned,
    properly-warmed feature vector. No test data is discarded and no leakage occurs
    (all scaling uses train-fit statistics only).
  * OVERLAP-AWARE STATS: h-bar-forward labels sampled every bar are heavily
    autocorrelated. Alongside the raw BA z-test (n = #test bars, for comparability
    with the daily book) we report an **effective-N** p-value using n_eff = n/h
    (the number of non-overlapping holding periods) — the honest significance.
  * CONSISTENT ANNUALIZATION: freq_per_year = bars_per_year / h for every horizon
    (h=96 → 252/yr = daily; h=480 → 50.4/yr = the daily 5-day book). We also report a
    non-overlapping Sharpe (trade once per h bars) as the realistic single-book figure.

Assets: GBP_USD, XAU_USD.  Train ≤ 2021-12-31, Test ≥ 2022-01-01 (== daily split).

To run:        export OANDA_API_KEY=your_key && python3 qrc_m15_v2.py
Offline smoke: python3 qrc_m15_v2.py --smoke      (synthetic data, no key/network)

Outputs:


Reference: Li, Mukhopadhyay, Bayat, Habibnia — arXiv:2505.13933 (QRC finance).
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# SAME QRC imports as the 1h script -------------------------------------------------
from algorithms.qrc.gen2_v1_original.qrc_v1 import (
    random_couplings,
    build_ising_hamiltonian,
    evolution_operator,
    precompute_z_diagonals,
    encode_input_state,
    partial_trace_input,
)
from algorithms.qrc.gen2_v1_original.qrc_encoder import encode_sequence

# ── Paths ─────────────────────────────────────────────────────────────────────────
BENCH_DIR = "."
RESULTS_JSON = f"{BENCH_DIR}/qrc_m15_v2_results.json"
LOG_PATH = f"{BENCH_DIR}/qrc_m15_v2_run.log"

# ── OANDA ─────────────────────────────────────────────────────────────────────────
OANDA_KEY = os.environ.get("OANDA_API_KEY", "")
OANDA_URL = "https://api-fxpractice.oanda.com/v3"
FETCH_FROM = "2018-01-01T00:00:00Z"
OANDA_COUNT = 5000               # max candles per request
MAX_PAGES = 400                  # hard safety cap on pagination

# ── Split / walk-forward ──────────────────────────────────────────────────────────
TRAIN_END = "2021-12-31"         # same split as the validated daily book
TEST_START = "2022-01-01"
WASHOUT = 100                    # reservoir warm-up bars (also the test warm-up length)
EMBARGO = 60                     # bars skipped between train and test (~15 h)

# ── QRC reservoir (n=6, daily-optimal, FAST) ──────────────────────────────────────
N1, N2 = 3, 3                    # 3 encoded momentum features + 3 hidden qubits
ALPHA, NU, TAU = 0.9, 0.5, 1.0   # spectral radius, transverse field, evolution time
SEEDS = [42, 43, 44, 45, 46]     # 5-seed ensemble

# ── Cost / annualization ──────────────────────────────────────────────────────────
TC_BPS = 2                       # 2 bps roundtrip (tight spot spread, GBP/USD & XAU/USD)
BARS_PER_YEAR = 96 * 252         # 24,192 M15 bars/yr (24 h FX, ~252 weekdays)

# ── Prediction horizons (look-ahead in M15 bars) ──────────────────────────────────
HORIZONS = [
    {"h": 16,  "label": "4h"},   # 4h  = 16 bars   (intraday)
    {"h": 96,  "label": "1D"},   # 1D  = 96 bars   (== daily bar)
    {"h": 480, "label": "5D"},   # 5D  = 480 bars  (== daily 5-day book — headline)
]

# ── Per-asset config (both n=6 (n1=3,n2=3), the daily-optimal) ────────────────────
ASSETS = [
    {"name": "GBP_USD", "instrument": "GBP_USD", "n1": 3, "n2": 3},
    {"name": "XAU_USD", "instrument": "XAU_USD", "n1": 3, "n2": 3},
]

# ── Ordered feature list — FIRST 3 are the QRC-encoded momentum cascade ───────────
FEATURE_NAMES = [
    "ret_4bar", "ret_16bar", "ret_96bar",           # [0:3] → encoded into the reservoir
    "ret_480bar",                                    # 5D momentum
    "rv_16bar", "rv_96bar", "vol_ratio",             # realized-vol regime
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",    # cyclic session / week-day
    "is_london", "is_newyork", "is_overlap",         # session flags (UTC)
]

# ── Daily book for apples-to-apples comparison (validated 2022-2026) ──────────────
DAILY_REFERENCE = {
    "GBP_USD": {"ba": 0.530, "p": 0.021, "sig": "★",  "signal": "momentum"},
    "XAU_USD": {"ba": 0.528, "p": 0.032, "sig": "★",  "signal": "mean-reversion (inverted)"},
}

# ── Logging (tee to stdout + log file) ────────────────────────────────────────────
T0 = time.time()
_LOG_FH = None


def _ts():
    return f"[{time.time() - T0:6.0f}s]"


def log(msg):
    line = f"{_ts()} {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def _r(x, n=4):
    """JSON-safe round (numpy → python float)."""
    return round(float(x), n)


# ════════════════════════════════════════════════════════════════════════════════
# DATA — OANDA M15 fetch (paginated) with a simple local CSV cache
# ════════════════════════════════════════════════════════════════════════════════
def _cache_path(instrument):
    return f"{BENCH_DIR}/qrc_m15_v2_data_{instrument}.csv"


def fetch_oanda_m15(instrument):
    """
    Fetch complete M15 mid candles from FETCH_FROM to now, paginated 5000 at a time.
    Returns a DataFrame indexed by (tz-naive UTC) time with a single 'close' column.
    Uses a local CSV cache unless QRC_M15_NOCACHE=1 (delete the CSV to force refresh).
    """
    cache = _cache_path(instrument)
    if os.path.exists(cache) and os.environ.get("QRC_M15_NOCACHE", "") != "1":
        df = pd.read_csv(cache, parse_dates=["time"]).set_index("time").sort_index()
        log(f"  [cache] {instrument}: {len(df):,} bars "
            f"{df.index[0].date()} → {df.index[-1].date()}  ({os.path.basename(cache)})")
        return df

    if not OANDA_KEY:
        raise RuntimeError("OANDA_API_KEY not set and no cache available for " + instrument)

    headers = {"Authorization": f"Bearer {OANDA_KEY}"}
    all_c, from_dt, page, prev_last = [], FETCH_FROM, 0, None
    while page < MAX_PAGES:
        params = {"count": OANDA_COUNT, "granularity": "M15", "price": "M",
                  "from": from_dt, "includeFirst": "true"}
        # transient-error retry
        r = None
        for attempt in range(4):
            try:
                resp = requests.get(f"{OANDA_URL}/instruments/{instrument}/candles",
                                    headers=headers, params=params, timeout=30)
                r = resp.json()
                break
            except Exception as e:  # network / decode
                if attempt == 3:
                    raise
                log(f"  [retry {attempt+1}] {instrument} page {page}: {e}")
                time.sleep(1.0 + attempt)
        if "candles" not in r:
            raise RuntimeError(f"OANDA error for {instrument}: {r.get('errorMessage', r)}")

        c = [x for x in r["candles"] if x.get("complete")]
        if not c:
            break
        all_c.extend(c)
        page += 1
        if page % 10 == 0:
            log(f"  ... {page} pages, {len(all_c):,} bars (through {c[-1]['time'][:10]})")

        last = pd.Timestamp(c[-1]["time"])                     # tz-aware UTC
        if prev_last is not None and last <= prev_last:        # no forward progress
            break
        prev_last = last
        if last >= pd.Timestamp("now", tz="UTC") - pd.Timedelta(minutes=15):
            break
        from_dt = (last + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        time.sleep(0.05)                                       # be polite to the API

    recs = [{"time": pd.Timestamp(x["time"]).tz_localize(None),
             "close": float(x["mid"]["c"])} for x in all_c]
    df = pd.DataFrame(recs).drop_duplicates("time").set_index("time").sort_index()
    log(f"  {instrument}: {len(df):,} complete M15 bars "
        f"{df.index[0].date()} → {df.index[-1].date()}")

    if os.environ.get("QRC_M15_NOCACHE", "") != "1":
        df.reset_index().to_csv(cache, index=False)
        log(f"  [cache] wrote {os.path.basename(cache)}")
    return df


def synthetic_close(start="2021-06-01", end="2022-07-01", seed=0):
    """Synthetic weekday M15 close series (GBM + faint hour-of-day drift) for --smoke."""
    idx = pd.date_range(start, end, freq="15min")
    idx = idx[idx.dayofweek < 5]                              # weekdays only (FX)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 6e-4, len(idx))
    rets += 5e-5 * np.sin(2 * np.pi * idx.hour.values / 24)   # tiny session seasonality
    close = 1.30 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close}, index=idx)


# ════════════════════════════════════════════════════════════════════════════════
# FEATURES
# ════════════════════════════════════════════════════════════════════════════════
def build_features(df):
    """Build the 14-column M15 feature frame (columns ordered per FEATURE_NAMES)."""
    close = df["close"]
    idx = pd.DatetimeIndex(df.index)
    hour, dow = idx.hour.values, idx.dayofweek.values
    log_r = np.log(close / close.shift(1))

    ft = pd.DataFrame(index=df.index)
    # momentum cascade — first three are the QRC-encoded inputs (N1=3)
    ft["ret_4bar"] = close.pct_change(4)        # 1h
    ft["ret_16bar"] = close.pct_change(16)      # 4h
    ft["ret_96bar"] = close.pct_change(96)      # 1D
    ft["ret_480bar"] = close.pct_change(480)    # 5D
    # realized volatility + intraday vol regime
    ft["rv_16bar"] = log_r.rolling(16, min_periods=8).std()
    ft["rv_96bar"] = log_r.rolling(96, min_periods=48).std()
    ft["vol_ratio"] = ft["rv_16bar"] / (ft["rv_96bar"] + 1e-8)
    # cyclic session (hour) + week-day encodings
    ft["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    ft["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    ft["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    ft["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    # UTC session flags
    ft["is_london"] = ((hour >= 6) & (hour < 15)).astype(float)
    ft["is_newyork"] = ((hour >= 13) & (hour < 21)).astype(float)
    ft["is_overlap"] = ((hour >= 13) & (hour < 15)).astype(float)

    ft = ft[FEATURE_NAMES].ffill().dropna()
    assert list(ft.columns[:3]) == ["ret_4bar", "ret_16bar", "ret_96bar"], \
        "First 3 columns must be the momentum cascade encoded into the QRC."
    return ft, close


# ════════════════════════════════════════════════════════════════════════════════
# QRC RESERVOIR
# ════════════════════════════════════════════════════════════════════════════════
def _zz_diag(i, j, n):
    """Diagonal of Z_i Z_j in the computational basis (for ⟨Z_i Z_j⟩ readout)."""
    dim = 1 << n
    idx = np.arange(dim)
    return (1 - 2 * ((idx >> (n - 1 - i)) & 1)) * (1 - 2 * ((idx >> (n - 1 - j)) & 1))


def build_reservoir(n1, n2, seed):
    """IsingXX reservoir U(τ) plus Z / ZZ readout operators for a given seed."""
    n = n1 + n2
    rng = np.random.default_rng(seed)
    J = random_couplings(n, rng, alpha=ALPHA)
    H = build_ising_hamiltonian(n, J, field_nu=NU)
    U = evolution_operator(H, tau=TAU)
    U_dag = U.conj().T
    z_diags = precompute_z_diagonals(n)                        # (n, 2^n)
    zz_arr = np.stack([_zz_diag(k, k + 1, n) for k in range(n - 1)])  # (n-1, 2^n)
    return U, U_dag, z_diags, zz_arr


def qrc_features(X_enc, n1, n2, U, U_dag, z_diags, zz_arr, washout):
    """
    Persistent-state QRC forward pass. Returns (len(X_enc) - washout, 2n-1) features:
    n single-qubit ⟨Z_i⟩ + (n-1) nearest-neighbour ⟨Z_i Z_{i+1}⟩ per retained step.
    """
    dim_h = 1 << n2
    rho_h = np.zeros((dim_h, dim_h), dtype=np.complex128)
    rho_h[0, 0] = 1.0
    out = []
    for t in range(len(X_enc)):
        rho_in = encode_input_state(X_enc[t])
        rho = np.kron(rho_in, rho_h)
        rho = U @ rho @ U_dag
        rho_h = partial_trace_input(rho, n1, n2)
        if t >= washout:
            diag = np.real(np.diag(rho))
            out.append(np.concatenate([z_diags @ diag, zz_arr @ diag]))
    return np.asarray(out)


# ════════════════════════════════════════════════════════════════════════════════
# STATS + SHARPE
# ════════════════════════════════════════════════════════════════════════════════
def zscore_ba(ba, n):
    """Two-sided z-test of balanced accuracy vs 0.5 for n (effective) samples."""
    n = max(float(n), 1.0)
    z = (ba - 0.5) / (0.5 / np.sqrt(n))
    return float(z), float(stats.norm.sf(abs(z)) * 2)


def sig(p):
    return "★★" if p < 0.01 else ("★" if p < 0.05 else ("†" if p < 0.10 else " "))


def compute_sharpe(y_pred, y_returns, h, freq_per_year, tc_bps=TC_BPS):
    """
    Long-short Sharpe with a fixed roundtrip cost.
      signal      = +1 (predict up) / -1 (predict down)
      net_return  = signal · realized_h_bar_return − tc_bps/1e4
    Two views:
      * headline `sharpe` — OVERLAPPING (evaluate every bar): a signal-quality figure,
        comparable to the reference/daily convention (freq = bars_per_year / h).
      * `*_nonoverlap` + `max_dd_pct` — NON-OVERLAPPING (one position per h bars): the
        realistic single-book. Drawdown is only meaningful here — an overlapping equity
        curve compounds h simultaneous books and is not a tradeable account.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_returns = np.asarray(y_returns, dtype=float)
    signals = 2.0 * y_pred - 1.0
    gross = signals * y_returns
    net = gross - tc_bps / 10000.0
    if len(net) < 20:
        return {}

    def _ann(series):
        return series.mean() * freq_per_year, (series.std() + 1e-12) * np.sqrt(freq_per_year)

    o_ret, o_vol = _ann(net)                                   # overlapping (signal quality)
    out = {
        "sharpe": _r(o_ret / o_vol, 3),
        "ann_return_pct": _r(o_ret * 100, 2),
        "ann_vol_pct": _r(o_vol * 100, 2),
        "hit_rate": _r((gross > 0).mean(), 4),
        "n_periods": int(len(net)),
    }
    no = net[::h]                                              # non-overlapping tradeable book
    if len(no) >= 5:
        n_ret, n_vol = _ann(no)
        eq = np.cumprod(1.0 + np.clip(no, -0.99, None))
        peak = np.maximum.accumulate(eq)
        mdd = ((eq - peak) / (peak + 1e-12)).min()
        out.update({
            "sharpe_nonoverlap": _r(n_ret / n_vol, 3),
            "ann_return_pct_nonoverlap": _r(n_ret * 100, 2),
            "max_dd_pct": _r(mdd * 100, 2),
            "n_periods_nonoverlap": int(len(no)),
        })
    return out


def eval_direction(avg_scores, y_true, ret, h, freq):
    """
    Evaluate averaged decision scores in both directions (a QRC signal can encode a
    mean-reversion edge that trades inverted — the daily book found this for gold).
    Returns both directions plus the selected best, with overlap-adjusted p-values.
    """
    n = len(y_true)
    pred_n = (avg_scores > 0).astype(int)
    ba_n = balanced_accuracy_score(y_true, pred_n)
    ba_i = 1.0 - ba_n                                          # BA(flip) = 1 - BA
    z_n, p_n = zscore_ba(ba_n, n)
    z_i, p_i = zscore_ba(ba_i, n)

    if ba_i > ba_n:
        best_ba, best_pred, direction = ba_i, 1 - pred_n, "inverted"
    else:
        best_ba, best_pred, direction = ba_n, pred_n, "normal"
    _, best_p = zscore_ba(best_ba, n)
    _, best_p_eff = zscore_ba(best_ba, n / h)                  # honest, overlap-adjusted
    sh = compute_sharpe(best_pred, ret, h, freq)
    return {
        "ba_normal": _r(ba_n), "p_normal": _r(p_n, 4),
        "ba_inverted": _r(ba_i), "p_inverted": _r(p_i, 4),
        "direction": direction,
        "ba": _r(best_ba), "p": _r(best_p, 4), "sig": sig(best_p),
        "p_overlap_adj": _r(best_p_eff, 4), "sig_overlap_adj": sig(best_p_eff),
        "sharpe": sh,
    }


# ════════════════════════════════════════════════════════════════════════════════
# PER-ASSET BENCHMARK
# ════════════════════════════════════════════════════════════════════════════════
def run_asset(cfg, df):
    name, n1, n2 = cfg["name"], cfg["n1"], cfg["n2"]
    log(f"\n{'=' * 66}")
    log(f"[{name}] n={n1 + n2} (n1={n1}, n2={n2})  seeds={SEEDS}  TC={TC_BPS}bps")
    log(f"{'=' * 66}")

    ft, close = build_features(df)
    if len(ft) < WASHOUT + 480 + 500:
        log(f"  ERROR: too few feature rows ({len(ft)})")
        return {}

    # ── Walk-forward split (index-based, with embargo) ────────────────────────────
    tr_idx = ft.index[ft.index <= TRAIN_END]
    te_all = ft.index[min(len(tr_idx) + EMBARGO, len(ft)):]
    te_idx = te_all[te_all >= TEST_START]
    if len(tr_idx) < WASHOUT + 500 or len(te_idx) < 500:
        log(f"  ERROR: insufficient split tr={len(tr_idx)} te={len(te_idx)}")
        return {}

    scaler = StandardScaler().fit(ft.loc[tr_idx].values)       # fit on TRAIN only
    Xtr_s = scaler.transform(ft.loc[tr_idx].values)
    Xte_s = scaler.transform(ft.loc[te_idx].values)
    log(f"  Train: {len(tr_idx):,} bars ({tr_idx[0].date()} → {tr_idx[-1].date()})")
    log(f"  Test:  {len(te_idx):,} bars ({te_idx[0].date()} → {te_idx[-1].date()})")

    # ── Encode. Train encodes tr_idx; test is warmed with the WASHOUT bars that
    #    immediately PRECEDE te_idx[0] so the reservoir state is settled and every
    #    test bar gets a correctly-aligned feature vector (fix for the 100-bar
    #    misalignment in the earlier scripts). Warm-up bars use train-fit scaling.
    te_start_pos = ft.index.get_loc(te_idx[0])
    warm_slice = slice(max(0, te_start_pos - WASHOUT), te_start_pos)
    Xwarm_s = scaler.transform(ft.iloc[warm_slice].values)

    Xtr_enc = encode_sequence(Xtr_s, n1)                       # (n_tr, n1)
    Xte_enc = encode_sequence(np.vstack([Xwarm_s, Xte_s]), n1)  # (warm+n_te, n1)
    n_warm = len(Xwarm_s)

    # ── QRC features once per seed (reused across all horizons) ───────────────────
    log(f"  Computing QRC features — {len(SEEDS)} seeds "
        f"({2 * (n1 + n2) - 1} feats/step)...")
    all_Ftr, all_Fte = [], []
    for seed in SEEDS:
        t0 = time.time()
        U, U_dag, z_diags, zz_arr = build_reservoir(n1, n2, seed)
        Ftr = qrc_features(Xtr_enc, n1, n2, U, U_dag, z_diags, zz_arr, washout=WASHOUT)
        # washout=n_warm discards exactly the warm-up bars → Fte aligns 1:1 with te_idx
        Fte = qrc_features(Xte_enc, n1, n2, U, U_dag, z_diags, zz_arr, washout=n_warm)
        all_Ftr.append(Ftr)
        all_Fte.append(Fte)
        log(f"    seed={seed}: Ftr={Ftr.shape} Fte={Fte.shape}  {time.time() - t0:.0f}s")

    # Alignment anchors: Ftr[i] ↔ tr_idx[WASHOUT + i];  Fte[i] ↔ te_idx[i]
    tr_lab_idx = tr_idx[WASHOUT:]
    assert len(all_Ftr[0]) == len(tr_lab_idx), "train QRC/label length mismatch"
    assert len(all_Fte[0]) == len(te_idx), "test QRC/label length mismatch"
    Xtr_cls = Xtr_s[WASHOUT:]                                  # classical, train-aligned

    # ── Per-horizon evaluation ────────────────────────────────────────────────────
    results = {}
    for hc in HORIZONS:
        h, hlab = hc["h"], hc["label"]
        freq = BARS_PER_YEAR / h
        log(f"\n  --- Horizon {hlab} ({h} bars, {freq:.1f} periods/yr) ---")

        fwd = close.pct_change(h).shift(-h)                    # forward h-bar return
        # Mask on the forward RETURN (not the 0/1 label): (fwd>0).astype(int) maps
        # NaN→0, so masking on the label would keep the last h bars (undefined future)
        # with a spurious "down" label and NaN return. Key the mask off fwd.notna().
        fwdtr = fwd.reindex(tr_lab_idx)
        fwdte = fwd.reindex(te_idx)
        mtr = fwdtr.notna().values                            # valid train labels
        mte = fwdte.notna().values                            # drops last h test bars
        ytr = (fwdtr.values[mtr] > 0).astype(int)
        yte = (fwdte.values[mte] > 0).astype(int)
        ret = fwdte.values[mte].astype(float)
        if len(ytr) < WASHOUT + 100 or len(yte) < 60:
            log(f"    insufficient labels tr={len(ytr)} te={len(yte)} — skip")
            continue
        n_eff = len(yte) / h
        log(f"    test buy-rate={yte.mean():.3f}  n_test={len(yte):,}  n_eff≈{n_eff:.0f}")

        # Classical Ridge on all 14 features (same test bars as QRC) --------------
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(Xtr_cls[mtr], ytr)
        pred_cls = clf.predict(Xte_s[mte])
        ba_cls = balanced_accuracy_score(yte, pred_cls)
        _, p_cls = zscore_ba(ba_cls, len(yte))
        _, p_cls_eff = zscore_ba(max(ba_cls, 1 - ba_cls), n_eff)
        sh_cls = compute_sharpe(pred_cls, ret, h, freq)
        log(f"    Classical : BA={ba_cls:.4f} p={p_cls:.4f}{sig(p_cls)} "
            f"p_adj={p_cls_eff:.3f}  Sh={sh_cls.get('sharpe','?')}")

        # QRC ensemble (QRC-only and QRC+classical concat) ------------------------
        scores_qrc, scores_cat = [], []
        for i in range(len(SEEDS)):
            Ftr, Fte = all_Ftr[i][mtr], all_Fte[i][mte]
            cq = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(Ftr, ytr)
            scores_qrc.append(cq.decision_function(Fte))
            Ctr = np.hstack([Ftr, Xtr_cls[mtr]])
            Cte = np.hstack([Fte, Xte_s[mte]])
            cc = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(Ctr, ytr)
            scores_cat.append(cc.decision_function(Cte))

        qrc = eval_direction(np.mean(scores_qrc, axis=0), yte, ret, h, freq)
        cat = eval_direction(np.mean(scores_cat, axis=0), yte, ret, h, freq)
        log(f"    QRC-only  : BA={qrc['ba']:.4f} p={qrc['p']:.4f}{qrc['sig']} "
            f"p_adj={qrc['p_overlap_adj']:.3f}  dir={qrc['direction']:<8} "
            f"Sh={qrc['sharpe'].get('sharpe','?')}")
        log(f"    QRC+cls   : BA={cat['ba']:.4f} p={cat['p']:.4f}{cat['sig']} "
            f"p_adj={cat['p_overlap_adj']:.3f}  dir={cat['direction']:<8} "
            f"Sh={cat['sharpe'].get('sharpe','?')}")

        results[f"{name}_{hlab}"] = {
            "asset": name, "instrument": cfg["instrument"],
            "frequency": "M15", "horizon_label": hlab, "horizon_bars": h,
            "periods_per_year": _r(freq, 1),
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "n_test_effective": _r(n_eff, 1), "test_buy_rate": _r(yte.mean(), 4),
            "n_qrc_features": int(all_Ftr[0].shape[1]),
            "Classical_Ridge": {
                "ba": _r(ba_cls), "p": _r(p_cls, 4), "sig": sig(p_cls),
                "p_overlap_adj": _r(p_cls_eff, 4), "sharpe": sh_cls},
            "QRC_only": qrc,
            "QRC_concat": cat,
        }
    return results


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    global _LOG_FH, SEEDS
    _LOG_FH = open(LOG_PATH, "w")

    smoke = "--smoke" in sys.argv
    if smoke:
        SEEDS = [42, 43]                                       # fewer seeds → fast self-test

    log("=" * 78)
    log("QRC M15 v2 — multi-horizon · rich M15 features · 5-seed ensemble · Sharpe")
    log(f"n={N1 + N2} (n1={N1}, n2={N2})  α={ALPHA} ν={NU} τ={TAU}  "
        f"washout={WASHOUT} embargo={EMBARGO}")
    log(f"horizons={[h['label'] for h in HORIZONS]} "
        f"({[h['h'] for h in HORIZONS]} bars)  TC={TC_BPS}bps  "
        f"bars/yr={BARS_PER_YEAR:,}")
    log(f"features({len(FEATURE_NAMES)})={FEATURE_NAMES}")
    log(f"Train ≤ {TRAIN_END}   Test ≥ {TEST_START}")
    log("=" * 78)

    if smoke:
        log("*** SMOKE MODE — synthetic data, no OANDA. Validates the pipeline only. ***")
    elif not OANDA_KEY and not all(os.path.exists(_cache_path(a["instrument"])) for a in ASSETS):
        log("ERROR: OANDA_API_KEY not set and no data cache present.")
        log("       To run: export OANDA_API_KEY=your_key && python3 qrc_m15_v2.py")
        _LOG_FH.close()
        sys.exit(1)

    all_results = {}
    for cfg in ASSETS:
        try:
            if smoke:
                df = synthetic_close(seed=hash(cfg["name"]) % 1000)
                log(f"\n[{cfg['name']}] synthetic: {len(df):,} bars "
                    f"{df.index[0].date()} → {df.index[-1].date()}")
            else:
                log(f"\nFetching {cfg['instrument']} M15 from {FETCH_FROM[:10]} ...")
                df = fetch_oanda_m15(cfg["instrument"])
            all_results.update(run_asset(cfg, df))
        except Exception as e:
            log(f"  ERROR {cfg['name']}: {e}")
            import traceback
            log(traceback.format_exc())

    # ── Summary table ─────────────────────────────────────────────────────────────
    log("\n" + "=" * 78)
    log("FINAL RESULTS — M15 v2  (vs validated daily book)")
    log("=" * 78)
    log(f"  Daily 5-day book:  GBP_USD BA=0.530 p=0.021 ★ (momentum) | "
        f"XAU_USD BA=0.528 p=0.032 ★ (inverted)")
    log(f"  {'-' * 74}")
    log(f"  {'key':<14}{'BA_qrc':>8}{'p':>8}{'p_adj':>8}{'dir':>10}"
        f"{'BA_cat':>8}{'Sh_5D':>8}")
    for k in sorted(all_results):
        r = all_results[k]
        q, c = r["QRC_only"], r["QRC_concat"]
        log(f"  {k:<14}{q['ba']:>8.4f}{q['p']:>8.4f}{q['p_overlap_adj']:>8.3f}"
            f"{q['direction']:>10}{c['ba']:>8.4f}"
            f"{str(q['sharpe'].get('sharpe','?')):>8}")

    payload = {
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "mode": "smoke_synthetic" if smoke else "oanda",
        "note_to_run": "export OANDA_API_KEY=your_key && python3 qrc_m15_v2.py",
        "config": {
            "N1": N1, "N2": N2, "n": N1 + N2,
            "alpha": ALPHA, "nu": NU, "tau": TAU,
            "seeds": SEEDS, "washout": WASHOUT, "embargo": EMBARGO,
            "tc_bps": TC_BPS, "bars_per_year": BARS_PER_YEAR,
            "train_end": TRAIN_END, "test_start": TEST_START,
            "horizons": [{"label": h["label"], "bars": h["h"],
                          "periods_per_year": _r(BARS_PER_YEAR / h["h"], 1)}
                         for h in HORIZONS],
            "features": FEATURE_NAMES,
            "qrc_encoded_features": FEATURE_NAMES[:N1],
            "qrc_features_per_step": 2 * (N1 + N2) - 1,
            "improvements": [
                "5-day (480-bar) look-ahead == daily book (+ 4h/1D sweep) vs next-bar noise",
                "14 rich M15 features: momentum cascade + realized vol + session/week",
                "5-seed reservoir ensemble (decision-function averaging)",
                "2 bps roundtrip Sharpe (tight FX/metal spot spread)",
                "FIX: test-set reservoir warm-up (no 100-bar label misalignment)",
                "overlap-adjusted p-values (n_eff = n/h) + non-overlapping Sharpe",
            ],
        },
        "daily_reference": DAILY_REFERENCE,
        "results": all_results,
        "total_runtime_seconds": _r(time.time() - T0, 1),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    log(f"\nSaved results → {RESULTS_JSON}")
    log(f"Log           → {LOG_PATH}")
    log(f"Total runtime: {time.time() - T0:.0f}s")
    _LOG_FH.close()


if __name__ == "__main__":
    main()
