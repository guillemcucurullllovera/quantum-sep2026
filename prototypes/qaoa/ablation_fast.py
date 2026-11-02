#!/usr/bin/env python3 -u
"""
Fast Ablation Sweep (Part A)
=============================
Re-runs the 3 ablations that timed out overnight, with reduced params
so they actually fit in the time budget.

What this answers:
  - Does Ledoit-Wolf shrinkage help or hurt?
  - Does momentum-based return estimator help or hurt?
  - Do both together help, or are they redundant?

Trimmed params (vs overnight sweep):
  - restarts:  3 (was 5)
  - maxiter:   100 (was 200)
  - q-values:  [0.5, 0.9] (was [0.3, 0.5, 0.7, 0.9])
  - Universe:  crypto10 (same as overnight)
  - Mode:      sharpe-opt (where shrinkage+momentum matter most)
  - p-layers:  2 (the sweet spot from overnight sweep)

Expected total runtime: ~30-45 min for all 3 configs.

HISTORICAL PROTOTYPE — preserved verbatim from the Gen1 research repository
(Gen1) phase0/scripts/ablation_fast.py. Orchestrates qaoa_backtest.py
(co-located in this directory) via subprocess with different CLI configs;
inherits that script's live-market-data dependency (see its header).
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).parent / "qaoa_backtest.py"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "ablation_fast"

CRYPTO10 = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
             "AVAX-USD", "LINK-USD", "ATOM-USD", "XRP-USD", "DOGE-USD"]

CONFIGS = [
    {
        "tag": "crypto10_sharpe_p2_baseline",  # Reference with both enhancements
        "flags": [],
    },
    {
        "tag": "crypto10_sharpe_p2_noshrink",
        "flags": ["--no-shrinkage"],
    },
    {
        "tag": "crypto10_sharpe_p2_nomomentum",
        "flags": ["--no-momentum"],
    },
    {
        "tag": "crypto10_sharpe_p2_noenhance",
        "flags": ["--no-shrinkage", "--no-momentum"],
    },
]


def run_config(config):
    json_out = str(RESULTS_DIR / f"{config['tag']}.json")
    log_file = str(RESULTS_DIR / f"{config['tag']}.log")

    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--assets", *CRYPTO10,
        "--period", "2y",
        "--interval", "1d",
        "--lookback-days", "20",
        "--rebalance-days", "5",
        "--cardinality", "5",
        "--p-layers", "2",
        "--restarts", "3",
        "--maxiter", "100",
        "--q-values", "0.5", "0.9",
        "--json-out", json_out,
    ] + config["flags"]

    print(f"\n{'─'*70}")
    print(f"ABLATION: {config['tag']}")
    print(f"Flags: {config['flags'] or 'none (baseline)'}")
    print(f"{'─'*70}")

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=2400)
    elapsed = time.time() - t0

    result = {"tag": config["tag"], "status": "OK" if proc.returncode == 0 else "FAIL",
              "time_s": round(elapsed, 1)}

    if os.path.exists(json_out):
        with open(json_out) as f:
            data = json.load(f)
        m = data.get("metrics", {}).get("qaoa", {})
        w = data.get("qaoa_win_rates", {})
        result.update({
            "qaoa_sharpe": m.get("sharpe"),
            "qaoa_return": m.get("total_return"),
            "win_vs_greedy": w.get("Greedy Forward"),
            "win_vs_sa": w.get("Sim. Annealing"),
        })
        print(f"  Sharpe: {m.get('sharpe'):+.3f}  Return: {m.get('total_return'):+.2%}  "
              f"vs Greedy: {w.get('Greedy Forward', '?')}  vs SA: {w.get('Sim. Annealing', '?')}")
    print(f"  Completed in {elapsed:.0f}s")
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("="*70)
    print("FAST ABLATION SWEEP (Part A)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configs: {len(CONFIGS)}")

    t0 = time.time()
    results = []
    for c in CONFIGS:
        try:
            results.append(run_config(c))
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
            results.append({"tag": c["tag"], "status": "TIMEOUT"})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"tag": c["tag"], "status": f"ERROR: {e}"})

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ABLATION COMPLETE ({total/60:.1f} min)")
    print(f"{'='*70}\n")

    print(f"  {'Config':<40} {'Sharpe':>8} {'Return':>9} {'vsGreedy':>9} {'vsSA':>9}")
    print(f"  {'-'*40} {'-'*8} {'-'*9} {'-'*9} {'-'*9}")
    for r in results:
        if r.get("qaoa_sharpe") is not None:
            print(f"  {r['tag']:<40} {r['qaoa_sharpe']:>+8.3f} "
                  f"{r.get('qaoa_return', 0):>+9.2%} "
                  f"{str(r.get('win_vs_greedy', '?')):>9} "
                  f"{str(r.get('win_vs_sa', '?')):>9}")
        else:
            print(f"  {r['tag']:<40} {r['status']}")

    # Save summary
    summary_file = str(RESULTS_DIR / "ablation_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_s": round(total, 1),
            "results": results,
        }, f, indent=2)
    print(f"\nSummary saved: {summary_file}")

    # Key finding
    baseline = next((r for r in results if "baseline" in r["tag"]
                     and r.get("qaoa_sharpe") is not None), None)
    if baseline:
        print(f"\nKey findings (vs baseline Sharpe {baseline['qaoa_sharpe']:+.3f}):")
        for r in results:
            if r == baseline or r.get("qaoa_sharpe") is None:
                continue
            delta = r["qaoa_sharpe"] - baseline["qaoa_sharpe"]
            label = r["tag"].split("_")[-1]
            verdict = "WORSE" if delta < 0 else "BETTER"
            print(f"  {label:<12}: Sharpe {r['qaoa_sharpe']:+.3f}  "
                  f"(Δ {delta:+.3f}) — {verdict}")


if __name__ == "__main__":
    main()
