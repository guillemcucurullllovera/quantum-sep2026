#!/usr/bin/env python3 -u
"""
N=15 Overnight Sweep — Production-Grade Validation
====================================================
Implements all 6 next-step features:
  1. Beat Equal Weight  → K=3 concentration + crypto bear universe + LONG-SHORT
  2. Transaction costs  → 20 bps per turnover unit on every run
  3. Bootstrap CI       → 95% confidence intervals on Sharpe
  4. Regime splits      → bull/bear/flat per-period analysis (built into backtest)
  5. Multi-seed         → 3 seeds per primary config, aggregated
  6. Scale beyond N=10  → N=15 universe across equity, crypto, mixed

Tighter params (vs failed N=20): 3 restarts, 100 maxiter, p=2.
Per-config runtime ~30-40 min. Total: ~7-9 hours overnight for ~13 runs.

HISTORICAL PROTOTYPE — preserved verbatim from the Gen1 research repository
(Gen1) phase0/scripts/n15_overnight.py. Orchestrates qaoa_backtest.py
(co-located in this directory) via subprocess; inherits its live-market-data
dependency (see that script's header).
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

SCRIPT = Path(__file__).parent / "qaoa_backtest.py"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "n15_overnight"

# ── Universes ──
EQUITY15 = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA",
            "GOOGL", "AMZN", "META", "TSLA", "AMD",
            "GLD", "TLT", "JPM", "V", "WMT"]
CRYPTO15 = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
            "AVAX-USD", "LINK-USD", "ATOM-USD", "XRP-USD", "DOGE-USD",
            "BNB-USD", "LTC-USD", "BCH-USD", "FIL-USD", "ETC-USD"]
MIXED15  = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
            "SPY", "QQQ", "AAPL", "NVDA", "GOOGL",
            "AMZN", "META", "GLD", "TLT", "JPM"]

# 20 bps tx cost on every run (industry default for retail)
TX_COST_BPS = 20.0
SEEDS_FOR_STABILITY = [7, 42, 101]  # 3 seeds for multi-seed configs


# ── Configs ──
# Each entry: (tag, universe_name, assets, k, p, mode, extra_flags, seeds)
# seeds=[None] means single deterministic run; seeds=SEEDS_FOR_STABILITY means multi-seed
CONFIGS = [
    # ───── PRIORITY 1 — Headline replication w/ multi-seed stability ─────
    ("equity_minvar_p2_k7", "equity15", EQUITY15, 7, 2, "min-var", [],
     SEEDS_FOR_STABILITY),

    # ───── PRIORITY 2 — Beat EW: concentrated K=3 ─────
    ("equity_minvar_p2_k3", "equity15", EQUITY15, 3, 2, "min-var", [],
     SEEDS_FOR_STABILITY),

    # ───── PRIORITY 3 — Beat EW: LONG-SHORT mode (sharpe-opt required) ─────
    ("equity_longshort_p2_k5", "equity15", EQUITY15, 5, 2, "sharpe",
     ["--long-short", "--q-values", "0.5", "0.9"],
     SEEDS_FOR_STABILITY),

    # ───── PRIORITY 4 — Universe where EW historically failed (crypto bear) ─────
    ("crypto_minvar_p2_k7", "crypto15", CRYPTO15, 7, 2, "min-var", [],
     SEEDS_FOR_STABILITY),

    # ───── PRIORITY 5 — Single-seed comparison configs ─────
    ("equity_minvar_p3_k7", "equity15", EQUITY15, 7, 3, "min-var", [], [7]),
    ("equity_minvar_p2_k10", "equity15", EQUITY15, 10, 2, "min-var", [], [7]),
    ("mixed_minvar_p2_k7", "mixed15", MIXED15, 7, 2, "min-var", [], [7]),
    ("equity_sharpe_p2_k7", "equity15", EQUITY15, 7, 2, "sharpe",
     ["--q-values", "0.5", "0.9"], [7]),

    # Long-short on crypto
    ("crypto_longshort_p2_k5", "crypto15", CRYPTO15, 5, 2, "sharpe",
     ["--long-short", "--q-values", "0.5", "0.9"], [7]),
]


def build_command(tag, assets, k, p, mode, extra_flags, seed):
    json_out = str(RESULTS_DIR / f"{tag}_seed{seed}.json")
    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--assets", *assets,
        "--period", "2y",
        "--interval", "1d",
        "--lookback-days", "20",
        "--rebalance-days", "10",   # 10-day rebalance cuts periods ~50%
        "--cardinality", str(k),
        "--p-layers", str(p),
        "--restarts", "2",          # was 3
        "--maxiter", "60",          # was 100
        "--no-momentum", "--no-shrinkage",   # ablation winner
        "--tx-cost-bps", str(TX_COST_BPS),
        "--bootstrap-ci",
        "--json-out", json_out,
    ] + extra_flags
    if mode == "min-var":
        cmd.append("--min-var")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd, json_out


def run_one(tag, universe, assets, k, p, mode, extra_flags, seed):
    cmd, json_out = build_command(tag, assets, k, p, mode, extra_flags, seed)
    log_file = str(RESULTS_DIR / f"{tag}_seed{seed}.log")
    print(f"\n  → seed={seed}  cmd={' '.join(cmd[7:])[:100]}...")

    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=5400)  # 90 min cap
        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {"seed": seed, "status": "TIMEOUT", "time_s": round(elapsed, 1)}

    res = {"seed": seed, "status": status, "time_s": round(elapsed, 1)}
    if os.path.exists(json_out):
        with open(json_out) as f:
            data = json.load(f)
        m = data.get("metrics", {})
        w = data.get("qaoa_win_rates", {})
        rg = data.get("qaoa_regime_split", {})
        ew_rg = data.get("ew_regime_split", {})
        q = m.get("qaoa", {})
        ew = m.get("equal_weight", {})
        res.update({
            "qaoa_sharpe": q.get("sharpe"),
            "qaoa_ci_lo": q.get("sharpe_ci_lo"),
            "qaoa_ci_hi": q.get("sharpe_ci_hi"),
            "qaoa_return": q.get("total_return"),
            "qaoa_dd": q.get("max_dd"),
            "ew_sharpe": ew.get("sharpe"),
            "ew_return": ew.get("total_return"),
            "win_vs_ew": w.get("Equal Weight (1/N)"),
            "win_vs_greedy": w.get("Greedy Forward"),
            "win_vs_sa": w.get("Sim. Annealing"),
            "regime_bull_qaoa": rg.get("bull", {}).get("sharpe"),
            "regime_bull_ew":   ew_rg.get("bull", {}).get("sharpe"),
            "regime_bear_qaoa": rg.get("bear", {}).get("sharpe"),
            "regime_bear_ew":   ew_rg.get("bear", {}).get("sharpe"),
            "regime_flat_qaoa": rg.get("flat", {}).get("sharpe"),
        })
    return res


def aggregate_seeds(per_seed_results):
    """Aggregate multi-seed results into mean ± std."""
    import numpy as np
    valid = [r for r in per_seed_results if r.get("qaoa_sharpe") is not None]
    if not valid:
        return None
    sharpes = [r["qaoa_sharpe"] for r in valid]
    returns = [r["qaoa_return"] for r in valid]
    win_ew  = [r["win_vs_ew"] for r in valid if r.get("win_vs_ew") is not None]
    return {
        "n_seeds": len(valid),
        "sharpe_mean": round(float(np.mean(sharpes)), 4),
        "sharpe_std": round(float(np.std(sharpes)), 4),
        "sharpe_min": round(float(np.min(sharpes)), 4),
        "sharpe_max": round(float(np.max(sharpes)), 4),
        "return_mean": round(float(np.mean(returns)), 4),
        "return_std": round(float(np.std(returns)), 4),
        "win_vs_ew_mean": round(float(np.mean(win_ew)), 4) if win_ew else None,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("="*78)
    print("N=15 OVERNIGHT SWEEP — Production-Grade Validation")
    print("="*78)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total_runs = sum(len(c[7]) for c in CONFIGS)
    print(f"Total individual runs: {total_runs} ({len(CONFIGS)} configs)")
    print()
    print("Features enabled on every run:")
    print(f"  • Transaction cost: {TX_COST_BPS} bps per turnover unit")
    print(f"  • Bootstrap 95% CI on Sharpe")
    print(f"  • Regime splits (bull/bear/flat)")
    print(f"  • Multi-seed stability where applicable (3 seeds)")
    print(f"  • Long-short mode where flagged (sharpe-opt only)")
    print(f"  • Concentrated K=3 to test 'beat EW'")
    print()

    t_start = time.time()
    all_results = []  # one row per (config, seed)
    by_config = defaultdict(list)

    for ci, (tag, universe, assets, k, p, mode, extra_flags, seeds) in enumerate(CONFIGS, 1):
        print(f"\n{'─'*78}")
        ls = " LONG-SHORT" if "--long-short" in extra_flags else ""
        print(f"[{ci}/{len(CONFIGS)}] CONFIG {tag}  ·  N=15 K={k} p={p} {mode}{ls}  ({len(seeds)} seed(s))")
        print(f"{'─'*78}")
        for seed in seeds:
            r = run_one(tag, universe, assets, k, p, mode, extra_flags, seed)
            r["tag"] = tag
            r["universe"] = universe
            r["k"] = k
            r["p"] = p
            r["mode"] = mode
            r["long_short"] = "--long-short" in extra_flags
            all_results.append(r)
            by_config[tag].append(r)
            sh = r.get("qaoa_sharpe")
            ew = r.get("ew_sharpe")
            if sh is not None:
                ci_str = ""
                if r.get("qaoa_ci_lo") is not None:
                    ci_str = f"  CI[{r['qaoa_ci_lo']:+.2f}, {r['qaoa_ci_hi']:+.2f}]"
                ew_str = f"  EW={ew:+.3f}" if ew is not None else ""
                wew = r.get("win_vs_ew")
                wew_str = f"  vsEW={wew}" if wew is not None else ""
                print(f"     ✓ seed={seed}  Sharpe={sh:+.3f}{ci_str}{ew_str}{wew_str}  ({r['time_s']/60:.1f}min)")
            else:
                print(f"     ✗ seed={seed}  {r.get('status')}  ({r['time_s']/60:.1f}min)")

        # Per-config aggregation if multi-seed
        agg = aggregate_seeds(by_config[tag])
        if agg and agg["n_seeds"] > 1:
            print(f"     ⊕ Aggregate ({agg['n_seeds']} seeds): "
                  f"Sharpe={agg['sharpe_mean']:+.3f} ± {agg['sharpe_std']:.3f}  "
                  f"[{agg['sharpe_min']:+.3f}, {agg['sharpe_max']:+.3f}]")

        elapsed = time.time() - t_start
        avg = elapsed / sum(len(by_config[t]) for t in by_config)
        remaining_runs = total_runs - sum(len(by_config[t]) for t in by_config)
        eta_h = (remaining_runs * avg) / 3600
        print(f"     ETA: {eta_h:.1f}h remaining")

    # ── FINAL REPORT ──
    total_time = time.time() - t_start
    print(f"\n{'='*78}")
    print(f"SWEEP COMPLETE  ·  {total_time/3600:.2f}h total")
    print(f"{'='*78}\n")

    # Leaderboard with aggregation per config
    print(f"  {'Config':<28} {'Sharpe (mean)':>15} {'± std':>8} {'EW':>7} {'vs EW':>8} {'95% CI':>17}")
    print(f"  {'-'*28} {'-'*15} {'-'*8} {'-'*7} {'-'*8} {'-'*17}")
    for tag, runs in by_config.items():
        agg = aggregate_seeds(runs)
        if agg is None:
            print(f"  {tag:<28}  FAILED")
            continue
        # average EW Sharpe across seeds
        ews = [r["ew_sharpe"] for r in runs if r.get("ew_sharpe") is not None]
        ew_avg = sum(ews)/len(ews) if ews else None
        # CI from first valid run
        first = next((r for r in runs if r.get("qaoa_ci_lo") is not None), None)
        ci_str = f"[{first['qaoa_ci_lo']:+.2f},{first['qaoa_ci_hi']:+.2f}]" if first else ""
        wew = agg.get("win_vs_ew_mean")
        wew_str = f"{wew:.1%}" if wew is not None else "—"
        print(f"  {tag:<28} {agg['sharpe_mean']:>+15.3f} {agg['sharpe_std']:>8.3f} "
              f"{ew_avg:>+7.3f} {wew_str:>8} {ci_str:>17}")

    # Save
    summary_file = str(RESULTS_DIR / "n15_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_hours": round(total_time/3600, 2),
            "tx_cost_bps": TX_COST_BPS,
            "results": all_results,
            "aggregated": {tag: aggregate_seeds(runs) for tag, runs in by_config.items()},
        }, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_file}")

    # Key questions
    print(f"\n{'─'*78}\nKEY QUESTIONS ANSWERED:\n{'─'*78}")
    import numpy as np

    # Q1: Multi-seed stability of headline config
    headline = aggregate_seeds(by_config.get("equity_minvar_p2_k7", []))
    if headline:
        print(f"  Q1 (Multi-seed stability):")
        print(f"     equity_minvar_p2_k7 across 3 seeds: "
              f"{headline['sharpe_mean']:+.3f} ± {headline['sharpe_std']:.3f}")
        print(f"     Range: [{headline['sharpe_min']:+.3f}, {headline['sharpe_max']:+.3f}]")
        if headline['sharpe_std'] < 0.1:
            print(f"     → STABLE result (std < 0.1)")
        else:
            print(f"     → noisy result (std > 0.1) — need more seeds")

    # Q2: Beat EW? Concentrated K=3
    k3 = aggregate_seeds(by_config.get("equity_minvar_p2_k3", []))
    if k3 and k3.get("win_vs_ew_mean") is not None:
        print(f"\n  Q2 (Concentrated K=3 vs EW):")
        print(f"     QAOA Sharpe (3 seeds): {k3['sharpe_mean']:+.3f}")
        print(f"     QAOA wins vs EW: {k3['win_vs_ew_mean']:.1%} of periods")
        if k3['win_vs_ew_mean'] > 0.5:
            print(f"     → BEATS EW per-period more than 50% — promising")
        else:
            print(f"     → still loses to EW most periods")

    # Q3: Long-short
    ls = aggregate_seeds(by_config.get("equity_longshort_p2_k5", []))
    if ls:
        print(f"\n  Q3 (Long-Short on equity):")
        print(f"     QAOA Sharpe: {ls['sharpe_mean']:+.3f} ± {ls['sharpe_std']:.3f}")

    # Q4: Crypto bear regime — does QAOA beat EW where EW historically lost?
    cr = aggregate_seeds(by_config.get("crypto_minvar_p2_k7", []))
    if cr and cr.get("win_vs_ew_mean") is not None:
        print(f"\n  Q4 (Crypto — does QAOA beat EW in bear regime?):")
        print(f"     QAOA Sharpe: {cr['sharpe_mean']:+.3f}")
        print(f"     QAOA wins vs EW: {cr['win_vs_ew_mean']:.1%}")

    # Q5: Tx-cost impact (compare to N=10 0.881 which was no-cost)
    if headline:
        print(f"\n  Q5 (Tx-cost impact):")
        print(f"     N=10 equity (NO tx cost): Sharpe +0.881")
        print(f"     N=15 equity (20 bps tx cost): {headline['sharpe_mean']:+.3f}")
        delta = headline['sharpe_mean'] - 0.881
        print(f"     Delta: {delta:+.3f}  (combination of scale + tx cost)")

    print(f"\nDone! All individual JSONs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
