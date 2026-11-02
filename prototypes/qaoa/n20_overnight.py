#!/usr/bin/env python3 -u
"""
N=20 Overnight Extension (Part B)
==================================
Tests whether QAOA's advantage emerges at larger problem sizes.

The search space at N=20, K=10 is C(20,10) = 184,756 portfolios.
Compare to N=10, K=5 which was only C(10,5) = 252 portfolios.

At N=20:
  - Classical exhaustive search still feasible but slow
  - Heuristics (greedy, SA, random) start missing good solutions
  - QAOA's structured quantum exploration should have more room to shine

Configs use the ABLATION-INFORMED defaults:
  - No momentum estimator (was hurting in bear markets)
  - Shrinkage optional (neutral impact)

What this answers:
  1. Does QAOA beat heuristics at N=20?
  2. Does QAOA's edge over classical exact shrink (as expected)?
  3. Is the edge better in min-var or sharpe-opt mode (post-ablation)?
  4. Does circuit depth matter more at larger N?
  5. Control: does the N=10 no-enhance result replicate?

HISTORICAL PROTOTYPE — preserved verbatim from the Gen1 research repository
(Gen1) phase0/scripts/n20_overnight.py. Orchestrates qaoa_backtest.py
(co-located in this directory) via subprocess; inherits its live-market-data
dependency (see that script's header).

Expected runtime: 6-10 hours for 9 configs.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).parent / "qaoa_backtest.py"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "n20_overnight"

# ── 20-asset universe (crypto + equity combined) ──
N20_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
    "AVAX-USD", "LINK-USD", "ATOM-USD", "XRP-USD", "DOGE-USD",
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
    "GOOGL", "AMZN", "META", "GLD", "TLT",
]

# ── 10-asset reference (for control) ──
CRYPTO10 = N20_UNIVERSE[:10]
EQUITY10 = N20_UNIVERSE[10:]

CONFIGS = [
    # ── Primary N=20 tests (ablation winner: BOTH off) ──
    {
        "tag": "n20_minvar_p2",
        "assets": N20_UNIVERSE, "k": 10, "p": 2, "mode": "min-var",
        "flags": ["--no-momentum", "--no-shrinkage"],  # ablation-winner default
    },
    {
        "tag": "n20_minvar_p3",
        "assets": N20_UNIVERSE, "k": 10, "p": 3, "mode": "min-var",
        "flags": ["--no-momentum", "--no-shrinkage"],
    },
    {
        "tag": "n20_minvar_p4",
        "assets": N20_UNIVERSE, "k": 10, "p": 4, "mode": "min-var",
        "flags": ["--no-momentum", "--no-shrinkage"],
    },
    {
        "tag": "n20_sharpe_p2_raw",  # raw stats, both off (ablation winner)
        "assets": N20_UNIVERSE, "k": 10, "p": 2, "mode": "sharpe",
        "flags": ["--no-momentum", "--no-shrinkage", "--q-values", "0.5", "0.9"],
    },
    # ── Comparison: does shrinkage help at N=20? (one with it ON) ──
    {
        "tag": "n20_minvar_p2_withshrink",
        "assets": N20_UNIVERSE, "k": 10, "p": 2, "mode": "min-var",
        "flags": ["--no-momentum"],  # shrinkage ON for comparison
    },
    # ── N=10 controls to verify ablation finding replicates ──
    {
        "tag": "n10_crypto_minvar_p2_noenhance",
        "assets": CRYPTO10, "k": 5, "p": 2, "mode": "min-var",
        "flags": ["--no-momentum", "--no-shrinkage"],
    },
    {
        "tag": "n10_equity_minvar_p2_noenhance",
        "assets": EQUITY10, "k": 5, "p": 2, "mode": "min-var",
        "flags": ["--no-momentum", "--no-shrinkage"],
    },
    # ── Smaller K: 5-of-20 (more concentrated) ──
    {
        "tag": "n20_k5_minvar_p2",
        "assets": N20_UNIVERSE, "k": 5, "p": 2, "mode": "min-var",
        "flags": ["--no-momentum", "--no-shrinkage"],
    },
    # ── Larger K: 15-of-20 (more diversified) ──
    {
        "tag": "n20_k15_minvar_p2",
        "assets": N20_UNIVERSE, "k": 15, "p": 2, "mode": "min-var",
        "flags": ["--no-momentum", "--no-shrinkage"],
    },
]


def build_command(config):
    json_out = str(RESULTS_DIR / f"{config['tag']}.json")
    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--assets", *config["assets"],
        "--period", "2y",
        "--interval", "1d",
        "--lookback-days", "20",
        "--rebalance-days", "5",
        "--cardinality", str(config["k"]),
        "--p-layers", str(config["p"]),
        "--restarts", "4",
        "--maxiter", "150",
        "--json-out", json_out,
    ] + config["flags"]
    if config["mode"] == "min-var":
        cmd.append("--min-var")
    return cmd, json_out


def run_config(config):
    cmd, json_out = build_command(config)
    log_file = str(RESULTS_DIR / f"{config['tag']}.log")
    n = len(config["assets"])

    print(f"\n{'─'*70}")
    print(f"CONFIG: {config['tag']}")
    print(f"  N={n}, K={config['k']}, p={config['p']}, mode={config['mode']}")
    print(f"  Flags: {config['flags'] or 'defaults'}")
    print(f"{'─'*70}")

    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {"tag": config["tag"], "status": "TIMEOUT", "time_s": round(elapsed, 1)}

    result = {"tag": config["tag"], "status": status, "time_s": round(elapsed, 1),
              "n_assets": n, "k": config["k"], "p": config["p"], "mode": config["mode"]}

    if os.path.exists(json_out):
        with open(json_out) as f:
            data = json.load(f)
        m = data.get("metrics", {})
        w = data.get("qaoa_win_rates", {})
        q = m.get("qaoa", {})
        cmv = m.get("classical_mv_exact", {})
        ew = m.get("equal_weight", {})

        result.update({
            "n_rebalances": data["config"]["n_rebalances"],
            "qaoa_sharpe": q.get("sharpe"),
            "qaoa_return": q.get("total_return"),
            "qaoa_maxdd": q.get("max_dd"),
            "cmv_sharpe": cmv.get("sharpe"),
            "ew_sharpe": ew.get("sharpe"),
            "win_vs_greedy": w.get("Greedy Forward"),
            "win_vs_random": w.get("Random Best-of-50"),
            "win_vs_sa": w.get("Sim. Annealing"),
            "win_vs_cmv": w.get("Classical MV (exact)"),
        })
        print(f"  QAOA: {q.get('sharpe'):+.3f} Sharpe, {q.get('total_return'):+.2%} return")
        print(f"  vs CMV: {w.get('Classical MV (exact)')}  vs Greedy: {w.get('Greedy Forward')}  "
              f"vs SA: {w.get('Sim. Annealing')}")
    print(f"  Completed in {elapsed:.0f}s [{status}]")
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("="*70)
    print("N=20 OVERNIGHT EXTENSION (Part B)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Universe (N=20): {', '.join(N20_UNIVERSE)}")
    print()
    print("Ablation-informed settings (DEFAULT: both enhancements OFF):")
    print("  - --no-momentum --no-shrinkage on primary runs (ablation winner)")
    print("  - One comparison run with shrinkage ON to verify at N=20")
    print("  - Reduced q-sweep: [0.5, 0.9] to prevent timeouts")
    print("  - 4 restarts, 150 iter (between ablation and overnight levels)")
    print()

    t0 = time.time()
    results = []
    for i, c in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Starting {c['tag']}...")
        try:
            results.append(run_config(c))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"tag": c["tag"], "status": f"ERROR: {e}"})

        avg_time = (time.time() - t0) / i
        remaining = (len(CONFIGS) - i) * avg_time
        print(f"  ETA: {remaining/3600:.1f}h remaining")

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"PART B COMPLETE ({total/3600:.2f}h)")
    print(f"{'='*70}\n")

    # ── Leaderboard ──
    ranked = sorted(
        [r for r in results if r.get("qaoa_sharpe") is not None],
        key=lambda r: r["qaoa_sharpe"], reverse=True
    )
    print(f"  {'Config':<35} {'N':>3} {'K':>3} {'Sharpe':>8} {'Return':>9} "
          f"{'vsCMV':>7} {'vsGrdy':>7} {'vsSA':>7}")
    print(f"  {'-'*35} {'-'*3} {'-'*3} {'-'*8} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")
    for r in ranked:
        print(f"  {r['tag']:<35} {r.get('n_assets', '?'):>3} {r.get('k', '?'):>3} "
              f"{r['qaoa_sharpe']:>+8.3f} {r.get('qaoa_return', 0):>+9.2%} "
              f"{str(r.get('win_vs_cmv', '?')):>7} "
              f"{str(r.get('win_vs_greedy', '?')):>7} "
              f"{str(r.get('win_vs_sa', '?')):>7}")

    # Save summary
    summary_file = str(RESULTS_DIR / "n20_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_hours": round(total/3600, 2),
            "results": results,
        }, f, indent=2)
    print(f"\nSummary: {summary_file}")

    # ── Key questions answered ──
    print(f"\n{'─'*70}")
    print("KEY QUESTIONS ANSWERED:")
    print(f"{'─'*70}")

    # Q1: Does QAOA beat heuristics at N=20?
    n20 = [r for r in results if r.get("n_assets") == 20
           and r.get("win_vs_greedy") is not None]
    if n20:
        import numpy as np
        avg_greedy = np.mean([r["win_vs_greedy"] for r in n20])
        avg_sa = np.mean([r["win_vs_sa"] for r in n20])
        print(f"  Q1: QAOA at N=20 vs Greedy: {avg_greedy:.1%} (vs 34.6% at N=10)")
        print(f"  Q1: QAOA at N=20 vs SA:     {avg_sa:.1%} (vs 39.0% at N=10)")

    # Q2: QAOA vs exact classical
    n20_cmv = [r["win_vs_cmv"] for r in results if r.get("n_assets") == 20
               and r.get("win_vs_cmv") is not None]
    if n20_cmv:
        import numpy as np
        print(f"  Q2: QAOA at N=20 vs exact CMV: {np.mean(n20_cmv):.1%} (lower is expected)")

    # Q3: min-var vs sharpe-opt at N=20
    n20_minvar = [r["qaoa_sharpe"] for r in results
                  if r.get("n_assets") == 20 and r.get("mode") == "min-var"
                  and r.get("qaoa_sharpe") is not None]
    n20_sharpe = [r["qaoa_sharpe"] for r in results
                  if r.get("n_assets") == 20 and r.get("mode") == "sharpe"
                  and r.get("qaoa_sharpe") is not None]
    if n20_minvar:
        import numpy as np
        print(f"  Q3: N=20 min-var avg Sharpe:    {np.mean(n20_minvar):+.3f}")
    if n20_sharpe:
        import numpy as np
        print(f"  Q3: N=20 sharpe-opt avg Sharpe: {np.mean(n20_sharpe):+.3f}")

    # Q4: depth scaling
    for p in [2, 3, 4]:
        p_sharpes = [r["qaoa_sharpe"] for r in results
                     if r.get("p") == p and r.get("n_assets") == 20
                     and r.get("qaoa_sharpe") is not None]
        if p_sharpes:
            import numpy as np
            print(f"  Q4: N=20 p={p} avg Sharpe: {np.mean(p_sharpes):+.3f}")

    # Q5: control — does ablation finding replicate at N=10?
    controls = [r for r in results if r["tag"].startswith("n10_")
                and r.get("qaoa_sharpe") is not None]
    if controls:
        print(f"  Q5: N=10 no-enhance controls:")
        for r in controls:
            print(f"     {r['tag']}: Sharpe {r['qaoa_sharpe']:+.3f}")

    print(f"\nDone! Individual results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
