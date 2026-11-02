#!/usr/bin/env python3 -u
"""
QAOA Overnight Sweep
=====================
Systematic parameter sweep that runs all meaningful configurations overnight.
Each run saves to a unique JSON file. A summary CSV is written at the end.

What this tests and WHY:
  - 3 universes: crypto-only, equities-only, mixed (cross-asset diversification)
  - 2 modes: min-variance vs sharpe-opt (does ignoring noisy mu help?)
  - 3 circuit depths: p=1,2,4 (does deeper circuit = better solution?)
  - 2 lookback windows: 15d, 30d (sensitivity to estimation window)
  - 2 rebalance frequencies: 5d, 10d (weekly vs biweekly)
  - With/without shrinkage and momentum as ablations

Total: ~30 runs. Each run ~10-30 min depending on N and periods.
Estimated total: 6-12 hours overnight.

All runs use --period 2y for maximum out-of-sample data.
Each rebalance is a genuine out-of-sample test: the optimizer only sees
trailing returns, then gets scored on future returns it never touched.

HISTORICAL PROTOTYPE — preserved verbatim from the Gen1 research repository
(Gen1) phase0/scripts/overnight_sweep.py. Orchestrates qaoa_backtest.py
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

SCRIPT = Path(__file__).parent / "qaoa_backtest.py"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "overnight_sweep"

# ── Asset universes ──
UNIVERSES = {
    "crypto10": {
        "assets": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
                    "AVAX-USD", "LINK-USD", "ATOM-USD", "XRP-USD", "DOGE-USD"],
        "period": "2y",
        "cardinality": 5,
    },
    "equity10": {
        "assets": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA",
                    "GOOGL", "AMZN", "META", "GLD", "TLT"],
        "period": "2y",
        "cardinality": 5,
    },
    "mixed10": {
        "assets": ["BTC-USD", "ETH-USD", "SOL-USD",
                    "SPY", "QQQ", "GLD", "TLT",
                    "AAPL", "MSFT", "NVDA"],
        "period": "1y",
        "cardinality": 5,
    },
}

# ── Parameter grid ──
# Core sweep: universe × mode × p_layers
CORE_CONFIGS = []
for univ_name, univ in UNIVERSES.items():
    for mode in ["min-var", "sharpe-opt"]:
        for p_layers in [1, 2, 4]:
            CORE_CONFIGS.append({
                "tag": f"{univ_name}_{mode}_p{p_layers}_lb20_rb5",
                "universe": univ_name,
                "mode": mode,
                "p_layers": p_layers,
                "lookback_days": 20,
                "rebalance_days": 5,
                "no_shrinkage": False,
                "no_momentum": False,
            })

# Sensitivity: lookback and rebalance on crypto with p=3
SENSITIVITY_CONFIGS = []
for lb in [15, 30]:
    for rb in [5, 10]:
        SENSITIVITY_CONFIGS.append({
            "tag": f"crypto10_min-var_p3_lb{lb}_rb{rb}",
            "universe": "crypto10",
            "mode": "min-var",
            "p_layers": 3,
            "lookback_days": lb,
            "rebalance_days": rb,
            "no_shrinkage": False,
            "no_momentum": False,
        })

# Ablations: test shrinkage and momentum impact on crypto
ABLATION_CONFIGS = [
    {
        "tag": "crypto10_sharpe_p3_noshrink",
        "universe": "crypto10",
        "mode": "sharpe-opt",
        "p_layers": 3,
        "lookback_days": 20,
        "rebalance_days": 5,
        "no_shrinkage": True,
        "no_momentum": False,
    },
    {
        "tag": "crypto10_sharpe_p3_nomomentum",
        "universe": "crypto10",
        "mode": "sharpe-opt",
        "p_layers": 3,
        "lookback_days": 20,
        "rebalance_days": 5,
        "no_shrinkage": False,
        "no_momentum": True,
    },
    {
        "tag": "crypto10_sharpe_p3_noenhance",
        "universe": "crypto10",
        "mode": "sharpe-opt",
        "p_layers": 3,
        "lookback_days": 20,
        "rebalance_days": 5,
        "no_shrinkage": True,
        "no_momentum": True,
    },
]

ALL_CONFIGS = CORE_CONFIGS + SENSITIVITY_CONFIGS + ABLATION_CONFIGS


def build_command(config):
    """Build the backtest command for a config."""
    univ = UNIVERSES[config["universe"]]
    json_out = str(RESULTS_DIR / f"{config['tag']}.json")

    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--assets"] + univ["assets"] + [
        "--period", univ["period"],
        "--interval", "1d",
        "--lookback-days", str(config["lookback_days"]),
        "--rebalance-days", str(config["rebalance_days"]),
        "--cardinality", str(univ["cardinality"]),
        "--p-layers", str(config["p_layers"]),
        "--restarts", "5",
        "--maxiter", "200",
        "--json-out", json_out,
    ]

    if config["mode"] == "min-var":
        cmd.append("--min-var")
    if config.get("no_shrinkage"):
        cmd.append("--no-shrinkage")
    if config.get("no_momentum"):
        cmd.append("--no-momentum")

    return cmd, json_out


def run_sweep():
    """Run all configurations sequentially, collecting results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("QAOA OVERNIGHT PARAMETER SWEEP")
    print("=" * 70)
    print(f"Total configurations: {len(ALL_CONFIGS)}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Print what we're testing
    print("Universes:")
    for name, u in UNIVERSES.items():
        print(f"  {name}: {len(u['assets'])} assets, K={u['cardinality']}, "
              f"period={u['period']}")
        print(f"    {', '.join(u['assets'])}")
    print()

    print("Sweep dimensions:")
    print(f"  Core: 3 universes × 2 modes × 3 depths (p=1,2,4) = {len(CORE_CONFIGS)} runs")
    print(f"  Sensitivity: 2 lookbacks × 2 rebalance freqs = {len(SENSITIVITY_CONFIGS)} runs")
    print(f"  Ablations: 3 feature toggles = {len(ABLATION_CONFIGS)} runs")
    print(f"  Total: {len(ALL_CONFIGS)} runs")
    print()

    results_summary = []
    sweep_start = time.time()

    for i, config in enumerate(ALL_CONFIGS):
        run_num = i + 1
        tag = config["tag"]
        print(f"\n{'─' * 70}")
        print(f"RUN {run_num}/{len(ALL_CONFIGS)}: {tag}")
        print(f"{'─' * 70}")

        cmd, json_out = build_command(config)
        log_file = str(RESULTS_DIR / f"{tag}.log")

        t0 = time.time()
        try:
            with open(log_file, "w") as lf:
                proc = subprocess.run(
                    cmd, stdout=lf, stderr=subprocess.STDOUT,
                    timeout=3600,  # 1 hour max per run
                )
            elapsed = time.time() - t0
            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"

            # Read the result JSON if it exists
            run_result = {"tag": tag, "status": status, "time_s": round(elapsed, 1)}
            if os.path.exists(json_out):
                with open(json_out) as f:
                    data = json.load(f)
                metrics = data.get("metrics", {})
                qaoa_m = metrics.get("qaoa", {})
                cmv_m = metrics.get("classical_mv_exact", {})
                greedy_m = metrics.get("greedy_forward", {})
                random_m = metrics.get("random_best_of_50", {})
                sa_m = metrics.get("simulated_annealing", {})
                ew_m = metrics.get("equal_weight", {})
                win_rates = data.get("qaoa_win_rates", {})

                run_result.update({
                    "n_rebalances": data["config"]["n_rebalances"],
                    "qaoa_sharpe": qaoa_m.get("sharpe"),
                    "qaoa_return": qaoa_m.get("total_return"),
                    "qaoa_maxdd": qaoa_m.get("max_dd"),
                    "cmv_sharpe": cmv_m.get("sharpe"),
                    "greedy_sharpe": greedy_m.get("sharpe"),
                    "random_sharpe": random_m.get("sharpe"),
                    "sa_sharpe": sa_m.get("sharpe"),
                    "ew_sharpe": ew_m.get("sharpe"),
                    "win_vs_greedy": win_rates.get("Greedy Forward"),
                    "win_vs_random": win_rates.get("Random Best-of-50"),
                    "win_vs_sa": win_rates.get("Sim. Annealing"),
                    "win_vs_cmv": win_rates.get("Classical MV (exact)"),
                })

                # Quick summary line
                print(f"  QAOA Sharpe: {qaoa_m.get('sharpe', '?'):>7} | "
                      f"Return: {qaoa_m.get('total_return', '?'):>+9} | "
                      f"vs Greedy: {win_rates.get('Greedy Forward', '?')}")
            else:
                print(f"  No results file generated")

            results_summary.append(run_result)
            print(f"  Completed in {elapsed:.0f}s [{status}]")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  TIMEOUT after {elapsed:.0f}s")
            results_summary.append({
                "tag": tag, "status": "TIMEOUT", "time_s": round(elapsed, 1)
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            results_summary.append({
                "tag": tag, "status": f"ERROR: {e}", "time_s": round(elapsed, 1)
            })

        # ETA
        avg_time = (time.time() - sweep_start) / run_num
        remaining = (len(ALL_CONFIGS) - run_num) * avg_time
        print(f"  ETA: {remaining/3600:.1f}h remaining "
              f"({run_num}/{len(ALL_CONFIGS)} done, "
              f"avg {avg_time:.0f}s/run)")

    # ── Write summary ──
    total_time = time.time() - sweep_start
    print(f"\n{'=' * 70}")
    print(f"SWEEP COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Successful: {sum(1 for r in results_summary if r['status'] == 'OK')}"
          f"/{len(results_summary)}")

    # Save summary JSON
    summary = {
        "sweep_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_hours": round(total_time / 3600, 2),
        "n_configs": len(ALL_CONFIGS),
        "results": results_summary,
    }
    summary_file = str(RESULTS_DIR / "sweep_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_file}")

    # Print leaderboard
    print(f"\n{'─' * 70}")
    print("LEADERBOARD (by QAOA Sharpe):")
    print(f"{'─' * 70}")
    ranked = sorted(
        [r for r in results_summary if r.get("qaoa_sharpe") is not None],
        key=lambda r: r["qaoa_sharpe"], reverse=True
    )
    print(f"  {'Config':<45} {'Sharpe':>7} {'Return':>9} {'vsGreedy':>9} {'vsSA':>9}")
    for r in ranked[:15]:
        print(f"  {r['tag']:<45} {r['qaoa_sharpe']:>7.3f} "
              f"{r.get('qaoa_return', 0):>+9.4f} "
              f"{r.get('win_vs_greedy', '?'):>9} "
              f"{r.get('win_vs_sa', '?'):>9}")

    # Print key findings
    print(f"\n{'─' * 70}")
    print("KEY QUESTIONS ANSWERED:")
    print(f"{'─' * 70}")

    # Best mode?
    minvar_sharpes = [r["qaoa_sharpe"] for r in ranked if "min-var" in r["tag"]
                      and r.get("qaoa_sharpe") is not None]
    sharpe_sharpes = [r["qaoa_sharpe"] for r in ranked if "sharpe" in r["tag"]
                      and r.get("qaoa_sharpe") is not None]
    if minvar_sharpes and sharpe_sharpes:
        import numpy as np
        print(f"  Min-var avg Sharpe:   {np.mean(minvar_sharpes):+.3f}")
        print(f"  Sharpe-opt avg Sharpe:{np.mean(sharpe_sharpes):+.3f}")

    # Best depth?
    for p in [1, 2, 4]:
        p_sharpes = [r["qaoa_sharpe"] for r in ranked if f"_p{p}_" in r["tag"]
                     and r.get("qaoa_sharpe") is not None]
        if p_sharpes:
            import numpy as np
            print(f"  p={p} avg Sharpe: {np.mean(p_sharpes):+.3f}")

    # Best universe?
    for u in UNIVERSES:
        u_sharpes = [r["qaoa_sharpe"] for r in ranked if r["tag"].startswith(u)
                     and r.get("qaoa_sharpe") is not None]
        if u_sharpes:
            import numpy as np
            print(f"  {u} avg Sharpe: {np.mean(u_sharpes):+.3f}")

    # QAOA vs heuristics overall
    all_greedy_wins = [r["win_vs_greedy"] for r in results_summary
                       if r.get("win_vs_greedy") is not None]
    all_sa_wins = [r["win_vs_sa"] for r in results_summary
                   if r.get("win_vs_sa") is not None]
    if all_greedy_wins:
        import numpy as np
        print(f"\n  QAOA avg win rate vs Greedy:  {np.mean(all_greedy_wins):.1%}")
    if all_sa_wins:
        import numpy as np
        print(f"  QAOA avg win rate vs SA:      {np.mean(all_sa_wins):.1%}")

    print(f"\nDone! Check {RESULTS_DIR} for all individual results.")


if __name__ == "__main__":
    run_sweep()
