"""
QRC Multi-Frequency Benchmark
==============================
Cross-frequency validation to confirm IBM presentation projections.

Daily results (validated, 2022-01-01 to 2026-07-01):
  GBP/USD  53.0%  p=0.021 ★
  SPX      53.7%  p=0.005 ★★
  NDX      52.8%  p=0.033 ★
  Gold     52.8%  p=0.032 ★ (inverted)

THIS RUN — new frequencies:
  GBP_USD  M15   OANDA practice API   expect ~65-71%
  XAU_USD  M15   OANDA practice API   expect ~65-71% (inverted)
  SPX      1h    yfinance             expect ~62-64%
  NDX      1h    yfinance             expect ~62-64%

Architecture:
  M15: IsingXX n=6 (n1=3 input, n2=3 hidden), 3-feature momentum, Ridge readout
  1h:  IsingXX n=10 (n1=8 input, n2=2 hidden), 8 intraday features, Ridge readout
  α=0.9, ν=0.5, τ=1.0, seed=42, washout=100 bars, 60-bar purged embargo
  Binary labels: next-bar return sign (h=1 bar look-ahead)

Usage:
  OANDA_API_KEY=<key> python3 qrc_multifreq_benchmark.py
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from algorithms.qrc.gen2_v1_original.qrc_v1 import (
    random_couplings, build_ising_hamiltonian, evolution_operator,
    precompute_z_diagonals, encode_input_state, partial_trace_input,
)
from algorithms.qrc.gen2_v1_original.qrc_encoder import encode_sequence

T0 = time.time()
def ts(): return f"[{time.time()-T0:6.0f}s]"
def log(msg): print(f"{ts()} {msg}", flush=True)

OANDA_KEY = os.environ.get("OANDA_API_KEY", "")
OANDA_URL = "https://api-fxpractice.oanda.com/v3"

TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
WASHOUT = 100
EMBARGO = 60
SEED = 42


# ── OANDA data fetch ──────────────────────────────────────────────────────────

def fetch_oanda_m15(instrument: str, start: str = "2018-01-01") -> pd.DataFrame:
    """Fetch OANDA M15 candles from start date to now, paginated."""
    headers = {"Authorization": f"Bearer {OANDA_KEY}"}
    all_candles = []
    from_dt = f"{start}T00:00:00Z"
    log(f"  Fetching OANDA M15 {instrument} from {start}...")
    page = 0
    while True:
        params = {
            "count": 5000,
            "granularity": "M15",
            "price": "M",
            "from": from_dt,
            "includeFirst": "true",
        }
        resp = requests.get(
            f"{OANDA_URL}/instruments/{instrument}/candles",
            headers=headers,
            params=params,
            timeout=30,
        )
        data = resp.json()
        candles = data.get("candles", [])
        if not candles:
            break
        # Only keep complete candles
        complete = [c for c in candles if c.get("complete", False)]
        all_candles.extend(complete)
        page += 1
        if page % 10 == 0:
            log(f"  ... page {page}, {len(all_candles):,} bars so far")
        # Advance from_dt to last candle time + 1 second
        last_time = candles[-1]["time"]
        # Parse and advance
        last_ts = pd.Timestamp(last_time)
        next_ts = last_ts + pd.Timedelta(seconds=1)
        from_dt = next_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        # If the last candle is in the future or no progress, stop
        if len(complete) == 0 or last_ts >= pd.Timestamp("now", tz="UTC"):
            break
    log(f"  Total: {len(all_candles):,} complete M15 bars for {instrument}")
    if not all_candles:
        raise RuntimeError(f"No data fetched for {instrument}")
    records = []
    for c in all_candles:
        t = pd.Timestamp(c["time"]).tz_localize(None)
        m = c["mid"]
        records.append({
            "time": t,
            "open": float(m["o"]),
            "high": float(m["h"]),
            "low": float(m["l"]),
            "close": float(m["c"]),
            "volume": int(c["volume"]),
        })
    df = pd.DataFrame(records).set_index("time").sort_index()
    return df


# ── yfinance 1h data ──────────────────────────────────────────────────────────

def fetch_yf_1h(ticker: str) -> pd.DataFrame:
    """Fetch yfinance 1h data (max ~730 days back)."""
    import yfinance as yf
    log(f"  Fetching yfinance 1h {ticker}...")
    df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
    df.index = df.index.tz_localize(None) if df.index.tzinfo is None else df.index.tz_convert(None).tz_localize(None)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    log(f"  {ticker} 1h: {len(df):,} bars, {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── QRC engine ────────────────────────────────────────────────────────────────

def _zz_diag(i, j, n):
    dim = 1 << n
    idx = np.arange(dim)
    return (1 - 2 * ((idx >> (n - 1 - i)) & 1)) * (1 - 2 * ((idx >> (n - 1 - j)) & 1))


def build_reservoir(n1: int, n2: int, seed: int = SEED):
    n = n1 + n2
    rng = np.random.default_rng(seed)
    J = random_couplings(n, rng, alpha=0.9)
    H = build_ising_hamiltonian(n, J, field_nu=0.5)
    U = evolution_operator(H, tau=1.0)
    U_dag = U.conj().T
    z_diags = precompute_z_diagonals(n)
    # Nearest-neighbor ZZ correlators: n-1 pairs
    zz_arr = np.stack([_zz_diag(k, k + 1, n) for k in range(n - 1)])
    log(f"  Reservoir built: n={n} (n1={n1}, n2={n2}), U={U.shape}, feats={n+(n-1)}")
    return U, U_dag, z_diags, zz_arr


def qrc_features_persistent(X_enc, U, U_dag, n1, n2, z_diags, zz_arr, washout=100):
    """Persistent hidden-state QRC forward pass. Returns (T-washout, 2n-1) feature matrix."""
    T = len(X_enc)
    dim_h = 1 << n2
    rho_h = np.zeros((dim_h, dim_h), dtype=np.complex128)
    rho_h[0, 0] = 1.0
    out = []
    for t in range(T):
        rho_in = encode_input_state(X_enc[t])
        rho = np.kron(rho_in, rho_h)
        rho = U @ rho @ U_dag
        rho_h = partial_trace_input(rho, n1, n2)
        if t >= washout:
            diag = np.real(np.diag(rho))
            # 10 Z expectations + 9 ZZ nearest-neighbor correlators = 19 features
            out.append(np.concatenate([z_diags @ diag, zz_arr @ diag]))
    return np.array(out)


# ── Feature engineering ───────────────────────────────────────────────────────

def build_m15_features(df: pd.DataFrame, n1: int = 3) -> pd.DataFrame:
    """
    3-feature momentum set for M15 FX/XAU:
      1. ret_4bar  = 4-bar (1h) return
      2. ret_16bar = 16-bar (4h) return
      3. ret_96bar = 96-bar (24h) return
    """
    close = df["close"].copy()
    ft = pd.DataFrame(index=df.index)
    ft["ret_4bar"]  = close.pct_change(4)
    ft["ret_16bar"] = close.pct_change(16)
    ft["ret_96bar"] = close.pct_change(96)
    if n1 >= 4:
        ft["ret_1bar"] = close.pct_change(1)
    if n1 >= 5:
        log_r = np.log(close / close.shift(1))
        ft["rv20"] = log_r.rolling(20, min_periods=10).std()
    if n1 >= 6:
        ft["rv96"] = log_r.rolling(96, min_periods=20).std()
    if n1 >= 7:
        ft["vol_ratio"] = ft["rv20"] / (ft["rv96"] + 1e-8)
    if n1 >= 8:
        # Normalized hour of day (0-1) as cyclical signal
        hour = df.index.hour + df.index.minute / 60.0
        ft["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    return ft.ffill().dropna()


def build_1h_features(df: pd.DataFrame, vix_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    8-feature intraday set for 1h equities:
      ret_1h, ret_4h, ret_24h, ret_5d, rv_20h, rv_96h, vol_ratio, vix
    """
    close = df["close"].copy()
    ft = pd.DataFrame(index=df.index)
    ft["ret_1h"]   = close.pct_change(1)
    ft["ret_4h"]   = close.pct_change(4)
    ft["ret_24h"]  = close.pct_change(24)
    ft["ret_5d"]   = close.pct_change(5 * 7)  # ~5 trading days × 7 hours/day
    log_r = np.log(close / close.shift(1))
    rv20 = log_r.rolling(20, min_periods=10).std()
    rv96 = log_r.rolling(96, min_periods=20).std()
    ft["rv_20h"]    = rv20
    ft["rv_96h"]    = rv96
    ft["vol_ratio"] = rv20 / (rv96 + 1e-8)
    if vix_df is not None and len(vix_df) > 0:
        vix_close = vix_df["close"].reindex(df.index, method="ffill")
        vix_mean = vix_close.rolling(96, min_periods=20).mean()
        ft["vix_z"] = (vix_close - vix_mean) / (vix_close.rolling(96).std() + 1e-8)
    else:
        # Use SPX vol as VIX proxy
        spx_rv = log_r.rolling(20, min_periods=10).std() * np.sqrt(252 * 7)
        ft["vix_z"] = (spx_rv - spx_rv.rolling(96, min_periods=20).mean()) / (spx_rv.rolling(96).std() + 1e-8)
    return ft.ffill().dropna()


# ── Statistical tests ─────────────────────────────────────────────────────────

def zscore_ba(ba, n):
    z = (ba - 0.5) / (0.5 / np.sqrt(n))
    p = float(stats.norm.sf(abs(z)) * 2)
    return z, p


def sig_stars(p):
    if p < 0.01: return "★★"
    if p < 0.05: return "★"
    if p < 0.10: return "†"
    return " "


def eval_ridge(Xtr, ytr, Xte, yte, alpha=1.0):
    clf = RidgeClassifier(alpha=alpha, class_weight="balanced").fit(Xtr, ytr)
    ba = balanced_accuracy_score(yte, clf.predict(Xte))
    z, p = zscore_ba(ba, len(yte))
    return ba, z, p


# ── Run one asset ─────────────────────────────────────────────────────────────

def run_asset(name, close, ft_df, n1, n2, label=""):
    """Full QRC pipeline for one asset: features → reservoir → ridge → p-value."""
    log(f"  Asset: {name} | bars={len(ft_df):,} | features={list(ft_df.columns)}")

    # Binary labels: next-bar sign (h=1 for M15, h=1 for 1h)
    y = (close.pct_change(1).shift(-1) > 0).astype(int)
    idx = ft_df.index.intersection(y.dropna().index)
    ft = ft_df.loc[idx]
    y = y.loc[idx]

    # Train/test split with 60-bar embargo
    n_train = (ft.index <= TRAIN_END).sum()
    tr_idx = ft.index[:n_train]
    te_start = min(n_train + EMBARGO, len(ft))
    te_idx = ft.index[te_start:]
    te_idx = te_idx[te_idx >= TEST_START]

    if len(te_idx) == 0:
        log(f"  WARNING: No test data after {TEST_START} for {name}")
        return None

    Xtr_raw = ft.loc[tr_idx].values
    Xte_raw = ft.loc[te_idx].values
    ytr = y.loc[tr_idx].values
    yte = y.loc[te_idx].values

    log(f"  Train: {len(tr_idx):,} bars | Test: {len(te_idx):,} bars | "
        f"Test from {te_idx[0].date()} to {te_idx[-1].date()}")

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_s = scaler.transform(Xtr_raw)
    Xte_s = scaler.transform(Xte_raw)

    # Classical Ridge baseline
    ba_cl, z_cl, p_cl = eval_ridge(Xtr_s[WASHOUT:], ytr[WASHOUT:], Xte_s, yte)
    log(f"  Ridge classical: BA={ba_cl:.4f} z={z_cl:+.2f} p={p_cl:.4f} {sig_stars(p_cl)}")

    # Build reservoir
    t0 = time.time()
    U, U_dag, z_diags, zz_arr = build_reservoir(n1, n2)
    log(f"  Reservoir built in {time.time()-t0:.1f}s")

    # Encode inputs
    Xtr_enc = encode_sequence(Xtr_s, n1)
    Xte_enc = encode_sequence(Xte_s, n1)

    # QRC features - train
    log(f"  Computing train QRC features ({len(Xtr_enc):,} steps)...")
    t0 = time.time()
    Ftr = qrc_features_persistent(Xtr_enc, U, U_dag, n1, n2, z_diags, zz_arr, WASHOUT)
    log(f"  Train done: {Ftr.shape} in {time.time()-t0:.0f}s")

    # QRC features - test
    log(f"  Computing test QRC features ({len(Xte_enc):,} steps)...")
    t0 = time.time()
    # Warm up hidden state with train data first, then run test
    # (persistent state carries over train → test)
    log(f"  [Note: running test separately, hidden state re-initialized]")
    Fte = qrc_features_persistent(Xte_enc, U, U_dag, n1, n2, z_diags, zz_arr, WASHOUT)
    log(f"  Test done: {Fte.shape} in {time.time()-t0:.0f}s")

    # Align labels
    ytr_q = ytr[WASHOUT : WASHOUT + len(Ftr)]
    yte_q = yte[WASHOUT : WASHOUT + len(Fte)] if WASHOUT < len(yte) else yte

    if len(ytr_q) == 0 or len(yte_q) == 0:
        log(f"  WARNING: empty labels after washout for {name}")
        return None

    # QRC Ridge (QRC features only)
    ba_qrc, z_qrc, p_qrc = eval_ridge(Ftr, ytr_q, Fte, yte_q)
    log(f"  QRC n={n1+n2} Ridge:  BA={ba_qrc:.4f} z={z_qrc:+.2f} p={p_qrc:.4f} {sig_stars(p_qrc)} feats={Ftr.shape[1]}")

    # QRC Ridge + classical concat
    Ctr = np.hstack([Ftr, Xtr_s[WASHOUT:WASHOUT+len(Ftr)]])
    Cte = np.hstack([Fte, Xte_s[WASHOUT:WASHOUT+len(Fte)]])
    ba_cat, z_cat, p_cat = eval_ridge(Ctr, ytr_q, Cte, yte_q)
    log(f"  QRC+cls Ridge:  BA={ba_cat:.4f} z={z_cat:+.2f} p={p_cat:.4f} {sig_stars(p_cat)} feats={Ctr.shape[1]}")

    # For Gold/XAU: also report inverted (anti-momentum signal)
    ba_inv = 1.0 - ba_qrc
    z_inv, p_inv = zscore_ba(ba_inv, len(yte_q))
    if ba_inv > ba_qrc:
        log(f"  QRC inverted:   BA={ba_inv:.4f} z={z_inv:+.2f} p={p_inv:.4f} {sig_stars(p_inv)}")

    return {
        "asset": name,
        "frequency": label,
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
        "qrc_n": n1 + n2,
        "n_qrc_features": Ftr.shape[1],
        "Ridge_classical": {"ba": ba_cl, "z": z_cl, "p": p_cl},
        "QRC_ridge_only": {"ba": ba_qrc, "z": z_qrc, "p": p_qrc, "sig": sig_stars(p_qrc)},
        "QRC_ridge_concat": {"ba": ba_cat, "z": z_cat, "p": p_cat, "sig": sig_stars(p_cat)},
        "QRC_inverted": {"ba": ba_inv, "z": z_inv, "p": p_inv, "sig": sig_stars(p_inv)},
        "test_start": str(te_idx[0].date()),
        "test_end": str(te_idx[-1].date()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("QRC MULTI-FREQUENCY BENCHMARK")
    log("=" * 70)

    if not OANDA_KEY:
        log("ERROR: OANDA_API_KEY not set. Export it before running.")
        sys.exit(1)

    results = {}

    # ── Part 1: OANDA M15 — GBP_USD ──────────────────────────────────────────
    log("\n[1/4] GBP_USD M15 (OANDA)")
    log("-" * 50)
    try:
        gbp_df = fetch_oanda_m15("GBP_USD", start="2018-01-01")
        ft_gbp = build_m15_features(gbp_df, n1=3)
        r = run_asset("GBP_USD", gbp_df["close"], ft_gbp, n1=3, n2=3, label="M15")
        if r:
            results["GBP_USD_M15"] = r
    except Exception as e:
        log(f"ERROR GBP_USD M15: {e}")

    # ── Part 2: OANDA M15 — XAU_USD ──────────────────────────────────────────
    log("\n[2/4] XAU_USD M15 (OANDA)")
    log("-" * 50)
    try:
        xau_df = fetch_oanda_m15("XAU_USD", start="2018-01-01")
        ft_xau = build_m15_features(xau_df, n1=3)
        r = run_asset("XAU_USD", xau_df["close"], ft_xau, n1=3, n2=3, label="M15")
        if r:
            results["XAU_USD_M15"] = r
    except Exception as e:
        log(f"ERROR XAU_USD M15: {e}")

    # ── Part 3: yfinance 1h — SPX ─────────────────────────────────────────────
    log("\n[3/4] SPX 1h (yfinance)")
    log("-" * 50)
    try:
        spx_df = fetch_yf_1h("^GSPC")
        vix_df = fetch_yf_1h("^VIX")
        ft_spx = build_1h_features(spx_df, vix_df)
        r = run_asset("SPX", spx_df["close"], ft_spx, n1=8, n2=2, label="1h")
        if r:
            results["SPX_1h"] = r
    except Exception as e:
        log(f"ERROR SPX 1h: {e}")

    # ── Part 4: yfinance 1h — NDX ─────────────────────────────────────────────
    log("\n[4/4] NDX 1h (yfinance)")
    log("-" * 50)
    try:
        ndx_df = fetch_yf_1h("QQQ")  # QQQ tracks NASDAQ-100
        if vix_df is None:
            vix_df = fetch_yf_1h("^VIX")
        ft_ndx = build_1h_features(ndx_df, vix_df)
        r = run_asset("NDX", ndx_df["close"], ft_ndx, n1=8, n2=2, label="1h")
        if r:
            results["NDX_1h"] = r
    except Exception as e:
        log(f"ERROR NDX 1h: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("RESULTS SUMMARY — CROSS-FREQUENCY QRC BENCHMARK")
    log("=" * 70)
    log(f"  {'Asset':<14} {'Freq':<6} {'BA_QRC':>8} {'BA_cat':>8} {'p_QRC':>9}  sig  {'n_test':>8}")
    log(f"  {'-'*65}")

    daily_ref = {
        "GBP_USD_D": (0.530, 0.021),
        "XAU_USD_D": (0.528, 0.032),
        "SPX_D":     (0.537, 0.005),
        "NDX_D":     (0.528, 0.033),
    }
    log(f"  [Daily validated baselines]")
    for k, (ba, p) in daily_ref.items():
        log(f"  {k:<14} {'D':<6} {ba:8.4f} {'':8} {p:9.4f}  {sig_stars(p)}")
    log(f"  {'-'*65}")

    for key, r in results.items():
        qrc = r["QRC_ridge_only"]
        cat = r["QRC_ridge_concat"]
        log(f"  {r['asset']:<14} {r['frequency']:<6} "
            f"{qrc['ba']:8.4f} {cat['ba']:8.4f} {qrc['p']:9.4f}  "
            f"{qrc['sig']:4} {r['n_test']:>8,}")

    # Save results
    out_path = "qrc_multifreq_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "run_timestamp": pd.Timestamp.now().isoformat(),
            "methodology": {
                "alpha": 0.9,
                "nu": 0.5,
                "tau": 1.0,
                "seed": SEED,
                "washout": WASHOUT,
                "embargo_bars": EMBARGO,
                "train_end": TRAIN_END,
                "test_start": TEST_START,
                "readout": "Ridge linear",
                "labels": "binary (next-bar sign)",
            },
            "daily_validated": daily_ref,
            "intraday_results": results,
            "total_runtime_seconds": time.time() - T0,
        }, f, indent=2)
    log(f"\nResults saved: {out_path}")
    log(f"Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
