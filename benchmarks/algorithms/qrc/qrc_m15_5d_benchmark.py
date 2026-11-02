"""
QRC M15 Benchmark — 5-day look-ahead (480-bar), same horizon as daily
=======================================================================
Uses 5-trading-day look-ahead (480 M15 bars) matching the daily benchmark's
h=5 (5 daily bars). This is the proper apples-to-apples cross-frequency test.

Daily validated: GBP/USD 53.0% p=0.021, XAU/USD 52.8% p=0.032 (inverted)
M15 expected:    if the QRC temporal memory is stronger at M15, BA should exceed daily.

Architecture:
  n=6 (n1=3, n2=3), α=0.9, ν=0.5, τ=1.0, seed=42
  3 momentum features: ret_4bar (1h), ret_16bar (4h), ret_96bar (24h)
  Walk-forward: train until 2021-12-31, test from 2022-01-01
  Embargo: 60 bars (= 60 M15 bars ≈ 15 hours)
  Look-ahead: h=480 bars (= 5 trading days × 96 M15 bars)
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
TRAIN_END = "2021-12-31"; TEST_START = "2022-01-01"
WASHOUT = 100; EMBARGO = 60
H_LOOKAHEAD = 480  # 5 trading days × 96 M15 bars/day
N1, N2 = 3, 3; SEED = 42


def fetch_oanda_m15(instrument):
    headers = {"Authorization": f"Bearer {OANDA_KEY}"}
    all_c = []; from_dt = "2018-01-01T00:00:00Z"; page = 0
    while True:
        r = requests.get(f"{OANDA_URL}/instruments/{instrument}/candles",
            headers=headers, params={"count":5000,"granularity":"M15","price":"M",
            "from":from_dt,"includeFirst":"true"}, timeout=30).json()
        c = [x for x in r.get("candles",[]) if x.get("complete")]
        if not c: break
        all_c.extend(c); page += 1
        if page % 10 == 0: log(f"  ... {page} pages, {len(all_c):,} bars")
        last = pd.Timestamp(c[-1]["time"])
        if last >= pd.Timestamp("now", tz="UTC"): break
        from_dt = (last + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recs = [{"time": pd.Timestamp(c["time"]).tz_localize(None),
             "close": float(c["mid"]["c"])} for c in all_c]
    df = pd.DataFrame(recs).set_index("time").sort_index()
    log(f"  {instrument}: {len(df):,} complete M15 bars")
    return df


def _zz_diag(i, j, n):
    dim = 1 << n; idx = np.arange(dim)
    return (1-2*((idx>>(n-1-i))&1)) * (1-2*((idx>>(n-1-j))&1))


def build_reservoir():
    n = N1 + N2; rng = np.random.default_rng(SEED)
    J = random_couplings(n, rng, alpha=0.9)
    H = build_ising_hamiltonian(n, J, field_nu=0.5)
    U = evolution_operator(H, tau=1.0); U_dag = U.conj().T
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


def zscore_ba(ba, n):
    z = (ba-0.5)/(0.5/np.sqrt(n)); return z, float(stats.norm.sf(abs(z))*2)

def sig(p):
    return "★★" if p<0.01 else ("★" if p<0.05 else ("†" if p<0.10 else " "))

def eval_ridge(Xtr, ytr, Xte, yte):
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(Xtr, ytr)
    ba = balanced_accuracy_score(yte, clf.predict(Xte))
    z, p = zscore_ba(ba, len(yte)); return ba, z, p


def run(name, instrument):
    log(f"\n{'='*60}")
    log(f"[{name}] M15 — 5-day (480-bar) look-ahead")
    log(f"{'='*60}")
    df = fetch_oanda_m15(instrument)
    close = df["close"]

    # Features: 3-bar momentum lookbacks
    ft = pd.DataFrame(index=df.index)
    ft["ret_4bar"]  = close.pct_change(4)
    ft["ret_16bar"] = close.pct_change(16)
    ft["ret_96bar"] = close.pct_change(96)
    ft = ft.ffill().dropna()

    # Labels: 5-trading-day look-ahead (480 M15 bars)
    y = (close.pct_change(H_LOOKAHEAD).shift(-H_LOOKAHEAD) > 0).astype(int)
    idx = ft.index.intersection(y.dropna().index)
    ft = ft.loc[idx]; y = y.loc[idx]

    n_tr = (ft.index <= TRAIN_END).sum()
    tr_idx = ft.index[:n_tr]
    te_idx = ft.index[min(n_tr+EMBARGO, len(ft)):]
    te_idx = te_idx[te_idx >= TEST_START]

    Xtr_raw = ft.loc[tr_idx].values; Xte_raw = ft.loc[te_idx].values
    ytr = y.loc[tr_idx].values; yte = y.loc[te_idx].values

    log(f"  Train: {len(tr_idx):,} bars | Test: {len(te_idx):,} bars")
    log(f"  Test: {te_idx[0].date()} → {te_idx[-1].date()}")
    log(f"  Class balance test: {yte.mean():.3f} buy")

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_s = scaler.transform(Xtr_raw); Xte_s = scaler.transform(Xte_raw)

    ba_cl, z_cl, p_cl = eval_ridge(Xtr_s[WASHOUT:], ytr[WASHOUT:], Xte_s, yte)
    log(f"  Ridge classical:  BA={ba_cl:.4f} p={p_cl:.4f} {sig(p_cl)}")

    log(f"  Building n=6 reservoir...")
    t0 = time.time()
    U, U_dag, z_diags, zz_arr = build_reservoir()
    log(f"  Reservoir built in {time.time()-t0:.1f}s")

    Xtr_enc = encode_sequence(Xtr_s, N1); Xte_enc = encode_sequence(Xte_s, N1)

    log(f"  Train QRC features ({len(Xtr_enc):,} steps)...")
    t0 = time.time()
    Ftr = qrc_features(Xtr_enc, U, U_dag, z_diags, zz_arr)
    log(f"  Train done: {Ftr.shape} in {time.time()-t0:.0f}s")

    log(f"  Test QRC features ({len(Xte_enc):,} steps)...")
    t0 = time.time()
    Fte = qrc_features(Xte_enc, U, U_dag, z_diags, zz_arr)
    log(f"  Test done: {Fte.shape} in {time.time()-t0:.0f}s")

    ytr_q = ytr[WASHOUT:WASHOUT+len(Ftr)]
    yte_q = yte[:len(Fte)]

    ba_qrc, z_qrc, p_qrc = eval_ridge(Ftr, ytr_q, Fte, yte_q)
    log(f"  QRC n=6 Ridge:    BA={ba_qrc:.4f} p={p_qrc:.4f} {sig(p_qrc)}")

    ba_inv, z_inv, p_inv = 1-ba_qrc, *zscore_ba(1-ba_qrc, len(yte_q))
    log(f"  QRC inverted:     BA={ba_inv:.4f} p={p_inv:.4f} {sig(p_inv)}")

    Ctr = np.hstack([Ftr, Xtr_s[WASHOUT:WASHOUT+len(Ftr)]])
    Cte = np.hstack([Fte, Xte_s[:len(Fte)]])
    ba_cat, z_cat, p_cat = eval_ridge(Ctr, ytr_q, Cte, yte_q)
    log(f"  QRC+cls Ridge:    BA={ba_cat:.4f} p={p_cat:.4f} {sig(p_cat)}")

    return {
        "asset": name, "instrument": instrument, "frequency": "M15",
        "lookahead_bars": H_LOOKAHEAD, "lookahead_label": "5-day",
        "n_train": len(tr_idx), "n_test": len(te_idx),
        "Ridge_classical":  {"ba": ba_cl,  "z": z_cl,  "p": p_cl,  "sig": sig(p_cl)},
        "QRC_ridge_only":   {"ba": ba_qrc, "z": z_qrc, "p": p_qrc, "sig": sig(p_qrc)},
        "QRC_inverted":     {"ba": ba_inv, "z": z_inv, "p": p_inv, "sig": sig(p_inv)},
        "QRC_ridge_concat": {"ba": ba_cat, "z": z_cat, "p": p_cat, "sig": sig(p_cat)},
    }


def main():
    log("=" * 60)
    log("QRC M15 BENCHMARK — 5-day look-ahead")
    log(f"H = {H_LOOKAHEAD} M15 bars ({H_LOOKAHEAD//96} trading days)")
    log("=" * 60)

    if not OANDA_KEY:
        log("ERROR: OANDA_API_KEY not set"); sys.exit(1)

    results = {}
    for name, inst in [("GBP_USD", "GBP_USD"), ("XAU_USD", "XAU_USD")]:
        try:
            r = run(name, inst)
            if r: results[name] = r
        except Exception as e:
            log(f"ERROR {name}: {e}")

    log("\n" + "=" * 60)
    log("RESULTS — M15 5-day look-ahead vs Daily")
    log("=" * 60)
    log(f"  {'Asset':<12} {'Freq':<6} {'BA_QRC':>8} {'BA_cat':>8} {'p_QRC':>9}  sig")
    log(f"  {'-'*55}")
    log("  Daily validated (GBP/USD inv for Gold):")
    log("  GBP_USD       D        0.5300            0.0210  ★")
    log("  XAU_USD       D        0.5280(inv)       0.0320  ★")
    log(f"  {'-'*55}")
    for k, r in results.items():
        qrc = r["QRC_ridge_only"]; cat = r["QRC_ridge_concat"]
        log(f"  {r['asset']:<12} {'M15 5d':<6} {qrc['ba']:8.4f} {cat['ba']:8.4f} {qrc['p']:9.4f}  {qrc['sig']}")

    out = "qrc_m15_5d_results.json"
    with open(out, "w") as f:
        json.dump({"results": results, "lookahead": "5-day (480 M15 bars)",
                   "total_runtime_seconds": time.time()-T0}, f, indent=2)
    log(f"\nSaved: {out}")
    log(f"Total: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
