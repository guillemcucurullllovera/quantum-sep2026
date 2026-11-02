"""
QRC 1h Benchmark — SPX and NDX
================================
yfinance only provides ~730 days of 1h data starting Aug 2023.
Adapted split: train 2023-08 to 2024-12, test 2025-01 to 2026-07.
n=10 (n1=8, n2=2), 8 intraday features, Ridge readout.
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
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

SEED = 42; WASHOUT = 100; EMBARGO = 60
# 1h data split (adjusted for yfinance 730d limit)
TRAIN_END_1H   = "2024-12-31"
TEST_START_1H  = "2025-01-01"
N1, N2 = 8, 2  # n=10 reservoir


def _zz_diag(i, j, n):
    dim = 1 << n; idx = np.arange(dim)
    return (1-2*((idx>>(n-1-i))&1)) * (1-2*((idx>>(n-1-j))&1))


def build_reservoir():
    n = N1 + N2
    rng = np.random.default_rng(SEED)
    J = random_couplings(n, rng, alpha=0.9)
    H = build_ising_hamiltonian(n, J, field_nu=0.5)
    U = evolution_operator(H, tau=1.0)
    U_dag = U.conj().T
    z_diags = precompute_z_diagonals(n)
    zz_arr = np.stack([_zz_diag(k, k+1, n) for k in range(n-1)])
    return U, U_dag, z_diags, zz_arr


def qrc_features(X_enc, U, U_dag, z_diags, zz_arr, washout=WASHOUT):
    dim_h = 1 << N2
    rho_h = np.zeros((dim_h, dim_h), dtype=np.complex128); rho_h[0,0]=1.0
    out = []
    for t in range(len(X_enc)):
        rho_in = encode_input_state(X_enc[t])
        rho = np.kron(rho_in, rho_h)
        rho = U @ rho @ U_dag
        rho_h = partial_trace_input(rho, N1, N2)
        if t >= washout:
            diag = np.real(np.diag(rho))
            out.append(np.concatenate([z_diags @ diag, zz_arr @ diag]))
    return np.array(out)


def fetch_and_build(ticker, vix_df):
    log(f"  Fetching {ticker} 1h...")
    df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
    df.index = df.index.tz_localize(None) if df.index.tzinfo is None else df.index.tz_convert(None).tz_localize(None)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    close = df["close"].copy()
    log(f"  {ticker}: {len(df):,} bars, {df.index[0].date()} → {df.index[-1].date()}")

    ft = pd.DataFrame(index=df.index)
    ft["ret_1h"]   = close.pct_change(1)
    ft["ret_4h"]   = close.pct_change(4)
    ft["ret_24h"]  = close.pct_change(24)
    ft["ret_5d"]   = close.pct_change(35)  # 5 trading days × 7h
    log_r = np.log(close / close.shift(1))
    rv20 = log_r.rolling(20, min_periods=10).std()
    rv96 = log_r.rolling(96, min_periods=20).std()
    ft["rv_20h"]    = rv20
    ft["rv_96h"]    = rv96
    ft["vol_ratio"] = rv20 / (rv96 + 1e-8)
    # VIX z-score
    vix_al = vix_df["close"].reindex(df.index, method="ffill")
    vix_mu = vix_al.rolling(96, min_periods=20).mean()
    vix_sd = vix_al.rolling(96, min_periods=20).std()
    ft["vix_z"] = (vix_al - vix_mu) / (vix_sd + 1e-8)
    ft = ft.ffill().dropna()

    # Binary labels: 1h look-ahead direction
    y = (close.pct_change(1).shift(-1) > 0).astype(int)
    idx = ft.index.intersection(y.dropna().index)
    ft = ft.loc[idx]; y = y.loc[idx]

    return ft, close, y


def zscore_ba(ba, n):
    z = (ba - 0.5) / (0.5 / np.sqrt(n))
    return z, float(stats.norm.sf(abs(z)) * 2)

def sig(p):
    return "★★" if p<0.01 else ("★" if p<0.05 else ("†" if p<0.10 else " "))

def eval_ridge(Xtr, ytr, Xte, yte):
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(Xtr, ytr)
    ba = balanced_accuracy_score(yte, clf.predict(Xte))
    z, p = zscore_ba(ba, len(yte))
    return ba, z, p


def run_1h(ticker, name, ft, close, y):
    log(f"\n  [{name}] bars={len(ft):,}")

    n_train = (ft.index <= TRAIN_END_1H).sum()
    tr_idx = ft.index[:n_train]
    te_start = min(n_train + EMBARGO, len(ft))
    te_idx = ft.index[te_start:]
    te_idx = te_idx[te_idx >= TEST_START_1H]

    if len(tr_idx) < WASHOUT + 50:
        log(f"  ERROR: Only {len(tr_idx)} training bars — insufficient")
        return None
    if len(te_idx) == 0:
        log(f"  ERROR: No test bars after {TEST_START_1H}")
        return None

    Xtr_raw = ft.loc[tr_idx].values; Xte_raw = ft.loc[te_idx].values
    ytr = y.loc[tr_idx].values; yte = y.loc[te_idx].values

    log(f"  Train: {len(tr_idx):,} bars ({tr_idx[0].date()} → {tr_idx[-1].date()})")
    log(f"  Test:  {len(te_idx):,} bars ({te_idx[0].date()} → {te_idx[-1].date()})")

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_s = scaler.transform(Xtr_raw); Xte_s = scaler.transform(Xte_raw)

    # Classical baseline
    ba_cl, z_cl, p_cl = eval_ridge(Xtr_s[WASHOUT:], ytr[WASHOUT:], Xte_s, yte)
    log(f"  Ridge classical:  BA={ba_cl:.4f} z={z_cl:+.2f} p={p_cl:.4f} {sig(p_cl)}")

    # Build reservoir and compute QRC features
    log(f"  Building n={N1+N2} reservoir...")
    t0 = time.time()
    U, U_dag, z_diags, zz_arr = build_reservoir()
    log(f"  Reservoir built in {time.time()-t0:.1f}s")

    Xtr_enc = encode_sequence(Xtr_s, N1)
    Xte_enc = encode_sequence(Xte_s, N1)

    log(f"  Computing train QRC features ({len(Xtr_enc):,} steps)...")
    t0 = time.time()
    Ftr = qrc_features(Xtr_enc, U, U_dag, z_diags, zz_arr)
    log(f"  Train done: {Ftr.shape} in {time.time()-t0:.0f}s")

    log(f"  Computing test QRC features ({len(Xte_enc):,} steps)...")
    t0 = time.time()
    Fte = qrc_features(Xte_enc, U, U_dag, z_diags, zz_arr)
    log(f"  Test done: {Fte.shape} in {time.time()-t0:.0f}s")

    ytr_q = ytr[WASHOUT:WASHOUT+len(Ftr)]
    yte_q = yte[:len(Fte)]

    # QRC only
    ba_qrc, z_qrc, p_qrc = eval_ridge(Ftr, ytr_q, Fte, yte_q)
    log(f"  QRC Ridge only:   BA={ba_qrc:.4f} z={z_qrc:+.2f} p={p_qrc:.4f} {sig(p_qrc)} feats={Ftr.shape[1]}")

    # QRC + classical concat
    Ctr = np.hstack([Ftr, Xtr_s[WASHOUT:WASHOUT+len(Ftr)]])
    Cte = np.hstack([Fte, Xte_s[:len(Fte)]])
    ba_cat, z_cat, p_cat = eval_ridge(Ctr, ytr_q, Cte, yte_q)
    log(f"  QRC+classical:    BA={ba_cat:.4f} z={z_cat:+.2f} p={p_cat:.4f} {sig(p_cat)} feats={Ctr.shape[1]}")

    return {
        "asset": name, "frequency": "1h", "ticker": ticker,
        "n_train": len(tr_idx), "n_test": len(te_idx),
        "n_qrc_features": Ftr.shape[1],
        "Ridge_classical":  {"ba": ba_cl,  "z": z_cl,  "p": p_cl,  "sig": sig(p_cl)},
        "QRC_ridge_only":   {"ba": ba_qrc, "z": z_qrc, "p": p_qrc, "sig": sig(p_qrc)},
        "QRC_ridge_concat": {"ba": ba_cat, "z": z_cat, "p": p_cat, "sig": sig(p_cat)},
        "train_start": str(tr_idx[0].date()), "train_end": str(tr_idx[-1].date()),
        "test_start":  str(te_idx[0].date()), "test_end":  str(te_idx[-1].date()),
    }


def main():
    log("=" * 70)
    log("QRC 1h BENCHMARK — SPX and NDX")
    log(f"Train: up to {TRAIN_END_1H} | Test: {TEST_START_1H}+")
    log("=" * 70)

    log("\nFetching ^VIX 1h...")
    vix_raw = yf.download("^VIX", period="730d", interval="1h", progress=False, auto_adjust=True)
    vix_raw.index = vix_raw.index.tz_localize(None) if vix_raw.index.tzinfo is None else vix_raw.index.tz_convert(None).tz_localize(None)
    vix_raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in vix_raw.columns]

    results = {}

    # SPX
    log("\n[1/2] S&P 500 (^GSPC)")
    ft_spx, close_spx, y_spx = fetch_and_build("^GSPC", vix_raw)
    r = run_1h("^GSPC", "SPX", ft_spx, close_spx, y_spx)
    if r: results["SPX_1h"] = r

    # NDX (via QQQ)
    log("\n[2/2] NASDAQ-100 (QQQ)")
    ft_ndx, close_ndx, y_ndx = fetch_and_build("QQQ", vix_raw)
    r = run_1h("QQQ", "NDX", ft_ndx, close_ndx, y_ndx)
    if r: results["NDX_1h"] = r

    # Summary
    log("\n" + "=" * 70)
    log("RESULTS — 1h CROSS-FREQUENCY BENCHMARK")
    log("=" * 70)
    log(f"  {'Asset':<12} {'Freq':<5} {'BA_cls':>8} {'BA_QRC':>8} {'BA_cat':>8} {'p_QRC':>9}  sig   n_test")
    log(f"  {'-'*75}")
    log("  Daily validated baselines:")
    log("  SPX           D        ---     0.5370     ---     0.0050  ★★")
    log("  NDX           D        ---     0.5280     ---     0.0330  ★ ")
    log(f"  {'-'*75}")
    for k, r in results.items():
        cl  = r["Ridge_classical"]
        qrc = r["QRC_ridge_only"]
        cat = r["QRC_ridge_concat"]
        log(f"  {r['asset']:<12} {'1h':<5} {cl['ba']:8.4f} {qrc['ba']:8.4f} {cat['ba']:8.4f} {qrc['p']:9.4f}  {qrc['sig']:4}  {r['n_test']:,}")

    out_path = "qrc_1h_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "run_timestamp": pd.Timestamp.now().isoformat(),
            "methodology": {"n": N1+N2, "n1": N1, "n2": N2, "alpha": 0.9, "nu": 0.5,
                            "tau": 1.0, "seed": SEED, "washout": WASHOUT,
                            "embargo": EMBARGO, "labels": "binary 1h look-ahead"},
            "results": results,
            "total_runtime_seconds": time.time() - T0,
        }, f, indent=2)
    log(f"\nResults saved: {out_path}")
    log(f"Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
