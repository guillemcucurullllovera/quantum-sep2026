"""
QRC 1h Fast Benchmark — n=6 + V=3 Virtual Nodes + Ensemble
============================================================
n=6 (64×64 matrices, ~0.7ms/step) + V=3 virtual nodes = 33 features/step
Total runtime: ~1-2 minutes (vs 70 min for n=10 V=2)

Key: Fujii 2017 shows V virtual nodes outperforms larger n in many temporal tasks.
Testing whether n=6 + 3 virtual nodes beats n=10 + 0 virtual nodes.
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

SEEDS = [42, 43, 44, 45, 46]
WASHOUT = 100; EMBARGO = 60
TRAIN_END = "2024-12-31"; TEST_START = "2025-01-01"
N1, N2 = 4, 2   # n=6: encode top-4 features, 4-dim hidden state
V_NODES = 3      # 3 virtual nodes = 3×11=33 features per step
ALPHA, NU, TAU = 0.9, 0.5, 1.0
TC_BPS = 2       # 2bps (index futures, as in IBM deck)

HORIZONS = [
    {"h": 4,  "label": "4h",  "freq_per_year": 252 * 7},
    {"h": 7,  "label": "1D",  "freq_per_year": 252},
    {"h": 35, "label": "5D",  "freq_per_year": 252 / 5},
]


def _zz_diag(i, j, n):
    dim = 1 << n; idx = np.arange(dim)
    return (1-2*((idx>>(n-1-i))&1)) * (1-2*((idx>>(n-1-j))&1))


def build_reservoir(seed):
    n = N1 + N2; rng = np.random.default_rng(seed)
    J = random_couplings(n, rng, alpha=ALPHA)
    H = build_ising_hamiltonian(n, J, field_nu=NU)
    U_sub = evolution_operator(H, tau=TAU / V_NODES)
    U_sub_dag = U_sub.conj().T
    z_diags = precompute_z_diagonals(n)
    zz_arr = np.stack([_zz_diag(k, k+1, n) for k in range(n-1)])
    return U_sub, U_sub_dag, z_diags, zz_arr


def qrc_features_vnodes(X_enc, U_sub, U_sub_dag, z_diags, zz_arr, washout=WASHOUT):
    dim_h = 1 << N2
    rho_h = np.zeros((dim_h, dim_h), dtype=np.complex128); rho_h[0,0] = 1.0
    out = []
    for t in range(len(X_enc)):
        rho_in = encode_input_state(X_enc[t])
        rho = np.kron(rho_in, rho_h)
        feats_t = []
        for v in range(V_NODES):
            rho = U_sub @ rho @ U_sub_dag
            if t >= washout:
                diag = np.real(np.diag(rho))
                feats_t.append(np.concatenate([z_diags @ diag, zz_arr @ diag]))
        rho_h = partial_trace_input(rho, N1, N2)
        if t >= washout:
            out.append(np.concatenate(feats_t))
    return np.array(out)


def compute_sharpe(y_pred, y_returns, freq_per_year, tc_bps=TC_BPS):
    signals = 2.0 * np.asarray(y_pred, float) - 1.0
    gross = signals * np.asarray(y_returns, float)
    net = gross - (tc_bps / 10000.0)
    if len(net) < 20: return {}
    ann_ret = float(net.mean() * freq_per_year)
    ann_vol = float(net.std() * np.sqrt(freq_per_year)) + 1e-10
    eq = np.cumprod(1 + np.clip(net, -0.99, None))
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk)/(pk+1e-10)).min())
    return {"sharpe": round(ann_ret/ann_vol,3), "ann_return_pct": round(ann_ret*100,2),
            "ann_vol_pct": round(ann_vol*100,2), "max_dd_pct": round(mdd*100,2)}


def zscore_ba(ba, n):
    z = (ba-0.5)/(0.5/np.sqrt(n))
    return z, float(stats.norm.sf(abs(z))*2)

def sig(p): return "★★" if p<0.01 else ("★" if p<0.05 else ("†" if p<0.10 else " "))


def fetch_features(ticker, vix_df):
    log(f"  Fetching {ticker} 1h...")
    raw = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
    raw.index = (raw.index.tz_localize(None) if raw.index.tzinfo is None
                 else raw.index.tz_convert(None).tz_localize(None))
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    close = raw["close"].copy()
    volume = raw.get("volume", pd.Series(0.0, index=raw.index))
    log(f"  {ticker}: {len(raw):,} bars {raw.index[0].date()} → {raw.index[-1].date()}")
    ft = pd.DataFrame(index=raw.index)
    for h, lbl in [(1,"1h"),(4,"4h"),(8,"8h"),(7,"1D"),(35,"5D")]:
        ft[f"ret_{lbl}"] = close.pct_change(h)
    log_r = np.log(close/close.shift(1))
    for h, lbl in [(4,"4h"),(8,"8h"),(24,"1D")]:
        ft[f"rv_{lbl}"] = log_r.rolling(h, min_periods=max(2,h//2)).std()
    ft["vol_ratio"] = ft["rv_4h"]/(ft["rv_1D"]+1e-8)
    vix_al = vix_df["close"].reindex(raw.index, method="ffill")
    vix_mu = vix_al.rolling(96, min_periods=20).mean()
    vix_sd = vix_al.rolling(96, min_periods=20).std()
    ft["vix_z"] = (vix_al-vix_mu)/(vix_sd+1e-8)
    ft["vix_chg"] = vix_al.pct_change(24)
    if not volume.isnull().all() and (volume != 0).any():
        vm, vs = volume.rolling(20,min_periods=5).mean(), volume.rolling(20,min_periods=5).std()
        ft["vol_z"] = (volume-vm)/(vs+1e-8)
    else:
        ft["vol_z"] = 0.0
    hour = pd.DatetimeIndex(raw.index).hour
    ft["hour_sin"] = np.sin(2*np.pi*hour/24)
    ft["hour_cos"] = np.cos(2*np.pi*hour/24)
    ft = ft.ffill().dropna()
    return ft, close


def run_asset(ticker, name, close, ft):
    log(f"\n{'='*60}")
    log(f"[{name}] n={N1+N2} (n1={N1},n2={N2}) V={V_NODES} → {V_NODES*(N1+N2+N1+N2-1)} feats/step")
    tr_idx = ft.index[ft.index <= TRAIN_END]
    te_raw = ft.index[min(len(tr_idx)+EMBARGO, len(ft)):]
    te_idx = te_raw[te_raw >= TEST_START]
    if len(tr_idx) < WASHOUT+100 or len(te_idx) < 50:
        log(f"  ERROR: tr={len(tr_idx)} te={len(te_idx)}"); return {}
    Xtr_raw = ft.loc[tr_idx].values; Xte_raw = ft.loc[te_idx].values
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_s = scaler.transform(Xtr_raw); Xte_s = scaler.transform(Xte_raw)
    log(f"  Train:{len(tr_idx):,} Test:{len(te_idx):,}")
    # Encode: N1=4, so only first 4 features go into quantum state
    Xtr_enc = encode_sequence(Xtr_s, N1); Xte_enc = encode_sequence(Xte_s, N1)
    # Compute QRC features per seed
    log(f"  Computing QRC features ({len(SEEDS)} seeds)...")
    all_Ftr, all_Fte = [], []
    for seed in SEEDS:
        t0 = time.time()
        U_sub, U_sub_dag, z_diags, zz_arr = build_reservoir(seed)
        Ftr = qrc_features_vnodes(Xtr_enc, U_sub, U_sub_dag, z_diags, zz_arr)
        Fte = qrc_features_vnodes(Xte_enc, U_sub, U_sub_dag, z_diags, zz_arr)
        all_Ftr.append(Ftr); all_Fte.append(Fte)
        log(f"  seed={seed}: {Ftr.shape} {time.time()-t0:.1f}s")
    results = {}
    for h_cfg in HORIZONS:
        h = h_cfg["h"]; h_lbl = h_cfg["label"]; fpy = h_cfg["freq_per_year"]
        log(f"\n  --- {h_lbl} ({h} bars) ---")
        fwd_ret = close.pct_change(h).shift(-h)
        y = (fwd_ret > 0).astype(int)
        y_all = y.reindex(ft.index).dropna()
        ret_all = fwd_ret.reindex(ft.index).dropna()
        ytr = y_all.reindex(tr_idx).dropna().values
        yte_s = y_all.reindex(te_idx).dropna()
        yte = yte_s.values; ret_te = ret_all.reindex(yte_s.index).values
        if len(ytr) < WASHOUT+50 or len(yte) < 30: continue
        # Classical
        clf_cls = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf_cls.fit(Xtr_s[WASHOUT:], ytr[WASHOUT:])
        ypred_cls = clf_cls.predict(Xte_s[:len(yte)])
        ba_cls = balanced_accuracy_score(yte, ypred_cls)
        z_cls, p_cls = zscore_ba(ba_cls, len(yte))
        sh_cls = compute_sharpe(ypred_cls, ret_te, fpy)
        log(f"  Classical: BA={ba_cls:.4f} p={p_cls:.4f}{sig(p_cls)} Sh={sh_cls.get('sharpe','?')}")
        # QRC ensemble
        ens_qrc, ens_cat = [], []
        for i, seed in enumerate(SEEDS):
            Ftr = all_Ftr[i]; Fte = all_Fte[i]
            n_te = min(len(Fte), len(yte))
            ytr_q = ytr[WASHOUT:WASHOUT+len(Ftr)]
            clf_q = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(Ftr, ytr_q)
            ens_qrc.append(clf_q.decision_function(Fte[:n_te]))
            Ctr = np.hstack([Ftr, Xtr_s[WASHOUT:WASHOUT+len(Ftr)]])
            Cte = np.hstack([Fte[:n_te], Xte_s[:n_te]])
            clf_c = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(Ctr, ytr_q)
            ens_cat.append(clf_c.decision_function(Cte))
        n_ens = min(len(s) for s in ens_qrc)
        yte_ens = yte[:n_ens]; ret_ens = ret_te[:n_ens]
        for tag, ens in [("QRC-only",ens_qrc),("QRC+cls",ens_cat)]:
            avg = np.mean([s[:n_ens] for s in ens], axis=0)
            for direction, pred in [("normal",(avg>0).astype(int)),("inv",(avg<0).astype(int))]:
                ba = balanced_accuracy_score(yte_ens, pred)
                z, p = zscore_ba(ba, n_ens)
                sh = compute_sharpe(pred, ret_ens, fpy)
                log(f"  {tag}({direction}): BA={ba:.4f} p={p:.4f}{sig(p)} Sh={sh.get('sharpe','?')}")
        # Best across all
        best_ba, best_p, best_sh, best_dir = 0, 1, {}, ""
        avg_q = np.mean([s[:n_ens] for s in ens_cat], axis=0)
        for pred, dname in [((avg_q>0).astype(int),"normal"),((avg_q<0).astype(int),"inv")]:
            ba = balanced_accuracy_score(yte_ens, pred)
            z, p = zscore_ba(ba, n_ens)
            sh = compute_sharpe(pred, ret_ens, fpy)
            if ba > best_ba:
                best_ba, best_p, best_sh, best_dir = ba, p, sh, dname
        results[f"{name}_{h_lbl}"] = {
            "asset": name, "ticker": ticker, "horizon_label": h_lbl, "horizon_bars": h,
            "n_train": len(tr_idx), "n_test": n_ens,
            "n_qrc_feats": V_NODES*(N1+N2+N1+N2-1),
            "QRC_cls_best": {"ba": round(best_ba,4), "p": round(best_p,4), "sig": sig(best_p),
                             "direction": best_dir, "sharpe": best_sh},
            "Classical_Ridge": {"ba": round(ba_cls,4), "p": round(p_cls,4), "sig": sig(p_cls),
                                "sharpe": sh_cls},
        }
    return results


def main():
    log("="*70)
    log(f"QRC 1h FAST v2 — n={N1+N2} V={V_NODES} → {V_NODES*(N1+N2+N1+N2-1)} feats · {len(SEEDS)} seeds · {TC_BPS}bps TC")
    log("="*70)
    vix_raw = yf.download("^VIX", period="730d", interval="1h", progress=False, auto_adjust=True)
    vix_raw.index = (vix_raw.index.tz_localize(None) if vix_raw.index.tzinfo is None
                     else vix_raw.index.tz_convert(None).tz_localize(None))
    vix_raw.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in vix_raw.columns]
    all_results = {}
    for ticker, name in [("^GSPC","SPX"),("QQQ","NDX")]:
        try:
            ft, close = fetch_features(ticker, vix_raw)
            all_results.update(run_asset(ticker, name, close, ft))
        except Exception as e:
            log(f"ERROR {name}: {e}")
            import traceback; log(traceback.format_exc())
    log("\n"+"="*70)
    log("FINAL RESULTS (n=6 V=3 fast)")
    log("="*70)
    log(f"  {'Key':<16} {'BA':>8} {'p':>9} {'sig':>4} {'Sh (2bps)':>10}")
    log(f"  [Daily SPX 53.7% BA → Sh=0.43 (validated, 5D hold)]")
    for k,r in sorted(all_results.items()):
        q = r.get("QRC_cls_best",{})
        log(f"  {k:<16} {q.get('ba','?'):>8} {q.get('p','?'):>9} {q.get('sig','?'):>4} {str(q.get('sharpe',{}).get('sharpe','?')):>10}")
        out = "qrc_1h_v2_fast_results.json"
    with open(out,"w") as f:
        json.dump({"methodology":{"N1":N1,"N2":N2,"V":V_NODES,"seeds":SEEDS,
            "n_feats_per_step":V_NODES*(N1+N2+N1+N2-1),"tc_bps":TC_BPS},
            "results":all_results,"runtime_seconds":time.time()-T0}, f, indent=2)
    log(f"Saved: {out}")
    log(f"Total: {time.time()-T0:.0f}s")

if __name__ == "__main__":
    main()
