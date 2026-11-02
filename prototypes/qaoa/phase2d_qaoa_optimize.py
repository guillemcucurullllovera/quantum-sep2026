#!/usr/bin/env python3 -u
"""
Phase 2D — QAOA Portfolio Optimization
========================================
End-to-end pipeline: fetch price data → compute covariance →
build QUBO → run QAOA → compare vs baselines → save results.

Usage:
  python scripts/phase2d_qaoa_optimize.py                          # Full 11-asset run
  python scripts/phase2d_qaoa_optimize.py --assets GBPUSD USDJPY SPY NVDA EURUSD
  python scripts/phase2d_qaoa_optimize.py --risk-aversion 0.7      # More conservative
  python scripts/phase2d_qaoa_optimize.py --p-layers 5              # Deeper circuit
  python scripts/phase2d_qaoa_optimize.py --cardinality 3           # Select 3 of N assets
  python scripts/phase2d_qaoa_optimize.py --validate                # Full comparison report
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# HISTORICAL PROTOTYPE — preserved from the Gen1 research repository,
# phase0/scripts/phase2d_qaoa_optimize.py. Import path updated to this
# repo's layout; algorithmic logic below is UNCHANGED from the original.
#
# NOT independently runnable in quantum_sep2026 as-is: the original module
# also exported `fetch_yahoo_prices` (live Yahoo Finance network fetch),
# `compute_covariance_matrix` (calls fetch_yahoo_prices internally), and
# `compute_expected_returns` / `adaptive_qubo_scaling` — none of which were
# carried into algorithms/qaoa_qubo/gen1/qaoa_portfolio.py during # port (that port dropped the network-coupled data-fetching layer entirely,
# per its own report). `adaptive_qubo_scaling` is pure math (not
# network-coupled) and appears to have been dropped incidentally rather
# than deliberately — worth restoring in a future pass if this script's
# `--validate` path needs it; not done here since fixing that is
# gen1/qaoa_portfolio.py's maintainers' call, not this prototype-import's.
from algorithms.qaoa_qubo.gen1.qaoa_portfolio import (
    PortfolioQUBO,
    QAOACircuit,
    QAOAOptimizer,
)

PHASE0_DIR = Path(__file__).parent.parent
RESULTS_DIR = PHASE0_DIR / "results"

DEFAULT_ASSETS = [
    "GBPUSD", "USDJPY", "EURUSD", "AUDUSD", "USDCAD", "USDCHF",
    "SPY", "NVDA", "AAPL", "MSFT", "TSLA",
]




# Annualization factor: mixed forex (24h) + equity (6.5h) assets
# Use a blended ~4000 trading hours/year as reasonable approximation
ANNUALIZE_HOURS = 4000  # sqrt(4000) ≈ 63.2


def annualize_sharpe(raw_sharpe):
    """Convert hourly Sharpe to annualized Sharpe."""
    return raw_sharpe * math.sqrt(ANNUALIZE_HOURS)


def run_classical_meanvar(mu, sigma, k):
    """
    Classical mean-variance: try all (N choose K) combinations,
    pick the one with best Sharpe. Equal-weight within selected assets.
    """
    from itertools import combinations
    n = len(mu)
    best_sharpe = -999
    best_x = None

    for combo in combinations(range(n), k):
        x = np.zeros(n)
        for i in combo:
            x[i] = 1.0 / k
        ret = mu @ x
        risk = np.sqrt(x @ sigma @ x)
        sharpe = ret / max(risk, 1e-10)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_x = np.zeros(n)
            for i in combo:
                best_x[i] = 1.0

    return best_x, best_sharpe


def run_equal_weight(mu, sigma, n):
    """Equal-weight (1/N) portfolio."""
    w = np.ones(n) / n
    ret = mu @ w
    risk = np.sqrt(w @ sigma @ w)
    sharpe = ret / max(risk, 1e-10)
    return sharpe


def run_random_portfolios(mu, sigma, k, n_samples=1000, seed=42):
    """Monte Carlo random K-of-N portfolios."""
    rng = np.random.RandomState(seed)
    n = len(mu)
    sharpes = []

    for _ in range(n_samples):
        indices = rng.choice(n, size=k, replace=False)
        w = np.zeros(n)
        w[indices] = 1.0 / k
        ret = mu @ w
        risk = np.sqrt(w @ sigma @ w)
        sharpe = ret / max(risk, 1e-10)
        sharpes.append(sharpe)

    return {
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "best_sharpe": float(np.max(sharpes)),
        "worst_sharpe": float(np.min(sharpes)),
        "p95_sharpe": float(np.percentile(sharpes, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2D — QAOA Portfolio Optimization")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--period", default="60d", help="Price data period (default: 60d)")
    parser.add_argument("--risk-aversion", type=float, default=0.5,
                        help="Risk aversion q ∈ [0,1]. Higher = more conservative.")
    parser.add_argument("--cardinality", type=int, default=None,
                        help="K: number of assets to hold. Default: N//2.")
    parser.add_argument("--p-layers", type=int, default=3,
                        help="QAOA circuit depth (default: 3)")
    parser.add_argument("--restarts", type=int, default=10,
                        help="Optimizer restarts (default: 10)")
    parser.add_argument("--maxiter", type=int, default=300,
                        help="Max optimizer iterations per restart (default: 300)")
    parser.add_argument("--mixer", choices=["xy", "rx"], default="xy",
                        help="Mixer type: xy (cardinality-preserving) or rx (standard)")
    parser.add_argument("--validate", action="store_true",
                        help="Run full comparison vs classical/random baselines")
    parser.add_argument("--json-out", default=None,
                        help="Output JSON path (default: results/phase2d_qaoa_results.json)")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 2D — QAOA PORTFOLIO OPTIMIZATION")
    print("=" * 60)
    print(f"Assets:      {len(args.assets)} ({', '.join(args.assets[:5])}{'...' if len(args.assets)>5 else ''})")
    print(f"Risk:        q={args.risk_aversion}")
    print(f"Circuit:     p={args.p_layers} layers, {args.mixer} mixer")
    print(f"Optimizer:   {args.restarts} restarts × {args.maxiter} iter")
    print(f"Data:        Yahoo {args.period}")

    # ── Fetch price data ──
    print(f"\n1. Fetching price data...")
    t0 = time.time()

    price_data = {}
    for asset in args.assets:
        print(f"  {asset}...", end=" ", flush=True)
        bars = fetch_yahoo_prices(asset, period=args.period, interval="1h")
        if bars and len(bars) >= 20:
            price_data[asset] = bars
            print(f"OK ({len(bars)} bars)")
        else:
            print("FAILED")

    valid_assets = [a for a in args.assets if a in price_data]
    n = len(valid_assets)
    print(f"  → {n} assets with valid data ({time.time()-t0:.1f}s)")

    if n < 3:
        print("ERROR: Need at least 3 assets")
        sys.exit(1)

    # ── Compute covariance and returns ──
    print(f"\n2. Computing covariance matrix ({n}×{n})...")
    cov_matrix, mean_returns, valid_assets = compute_covariance_matrix(
        valid_assets, price_data=price_data
    )
    n = len(valid_assets)

    print(f"  Assets: {', '.join(valid_assets)}")
    print(f"  Mean hourly returns (raw):")
    for i, a in enumerate(valid_assets):
        print(f"    {a:>8}: {mean_returns[i]:>+.6f}")

    # Covariance condition number
    eigvals = np.linalg.eigvalsh(cov_matrix)
    cond = eigvals[-1] / max(eigvals[0], 1e-15)
    print(f"  Cov condition number: {cond:.1f}")

    # ── Setup QUBO ──
    k = args.cardinality or max(2, n // 2)
    k = min(k, n - 1)

    # With XY-mixer: no penalty needed (mixer preserves cardinality)
    penalty = 0.0 if args.mixer == "xy" else 2.0

    # Adaptive scaling: automatically balance return vs risk in QUBO
    mu_scaled, cov_scaled = adaptive_qubo_scaling(mean_returns, cov_matrix, k,
                                                   target_return_frac=0.4)
    print(f"\n3. Building QUBO (K={k} of {n}, λ={penalty}, adaptive scaling)...")
    print(f"  Return scale: {np.mean(np.abs(mu_scaled))/max(np.mean(np.abs(mean_returns)),1e-20):.1f}x")
    print(f"  Cov scale: {np.mean(np.abs(cov_scaled))/max(np.mean(np.abs(cov_matrix)),1e-20):.1f}x")
    qubo = PortfolioQUBO(
        expected_returns=mu_scaled,
        cov_matrix=cov_scaled,
        risk_aversion=args.risk_aversion,
        cardinality_k=k,
        penalty_lambda=penalty,
        asset_names=valid_assets,
    )

    J, h, offset = qubo.build_ising()
    print(f"  Ising: max|J|={np.max(np.abs(J)):.4f}, max|h|={np.max(np.abs(h)):.4f}")

    # ── Brute force (ground truth) ──
    # With XY-mixer, only search K-hot bitstrings (the feasible set)
    from math import comb as _comb
    n_feasible = _comb(n, k)
    if n <= 16:
        print(f"\n4. Brute force ({n_feasible:,} K-hot portfolios, K={k})...", end=" ", flush=True)
        t0 = time.time()
        bf_x, bf_cost, bf_top = qubo.classical_bruteforce(constrain_k=True)
        print(f"done ({time.time()-t0:.1f}s)")
        # Evaluate on RAW data so Sharpe is comparable with baselines
        raw_qubo_bf = PortfolioQUBO(
            expected_returns=mean_returns, cov_matrix=cov_matrix,
            risk_aversion=args.risk_aversion, cardinality_k=k,
            penalty_lambda=0.0, asset_names=valid_assets,
        )
        bf_portfolio = raw_qubo_bf.evaluate_portfolio(bf_x, annualize_hours=ANNUALIZE_HOURS)
        bf_allocation = raw_qubo_bf.decode_bitstring(bf_x)
        print(f"  Optimal: {' '.join(str(int(b)) for b in bf_x)}")
        print(f"  Assets: {', '.join(bf_allocation.keys())}")
        print(f"  Sharpe (hourly): {bf_portfolio['sharpe']:.4f}")
        print(f"  Sharpe (annual): {bf_portfolio['sharpe_annual']:.4f}")
    else:
        bf_x, bf_cost, bf_top = None, None, None
        bf_portfolio = None
        print(f"\n4. Brute force: skipped (N={n} too large)")

    # ── QAOA optimization ──
    print(f"\n5. QAOA optimization (p={args.p_layers}, {args.mixer}-mixer, {args.restarts} restarts)...")
    t0 = time.time()

    circuit = QAOACircuit(
        n_assets=n, J=J, h=h,
        p_layers=args.p_layers,
        mixer=args.mixer,
        cardinality_k=k,
    )

    optimizer = QAOAOptimizer(
        circuit, qubo,
        maxiter=args.maxiter,
        n_restarts=args.restarts,
    )

    qaoa_result = optimizer.optimize(verbose=True)
    qaoa_time = time.time() - t0

    # Evaluate on RAW data so Sharpe is comparable with baselines
    raw_qubo = PortfolioQUBO(
        expected_returns=mean_returns, cov_matrix=cov_matrix,
        risk_aversion=args.risk_aversion, cardinality_k=k,
        penalty_lambda=0.0, asset_names=valid_assets,
    )
    qaoa_x = np.array([int(b) for b in qaoa_result["best_bitstring"]], dtype=float)
    qaoa_portfolio = raw_qubo.evaluate_portfolio(qaoa_x, annualize_hours=ANNUALIZE_HOURS)
    qaoa_allocation = raw_qubo.decode_bitstring(qaoa_x)

    print(f"\n  QAOA result ({qaoa_time:.1f}s):")
    print(f"  Bitstring: {qaoa_result['best_bitstring']}  (prob={qaoa_result['best_probability']:.4f})")
    print(f"  Assets: {', '.join(qaoa_allocation.keys())}")
    print(f"  Sharpe (hourly): {qaoa_portfolio['sharpe']:.4f}")
    print(f"  Sharpe (annual): {qaoa_portfolio['sharpe_annual']:.4f}")
    print(f"  Expected return: {qaoa_portfolio['expected_return']:.6f}")
    print(f"  Risk: {qaoa_portfolio['risk']:.6f}")

    if bf_portfolio:
        print(f"  vs Brute force (annual): {qaoa_portfolio['sharpe_annual']:.4f} vs {bf_portfolio['sharpe_annual']:.4f}")
        if bf_portfolio['sharpe'] != 0:
            ratio = qaoa_portfolio['sharpe'] / bf_portfolio['sharpe']
            print(f"  Approximation ratio: {ratio:.4f}")

    # ── Baselines ──
    print(f"\n6. Baselines...")

    # Baselines use raw data (Sharpe = ret/risk is scale-invariant)
    # Equal weight
    ew_sharpe = run_equal_weight(mean_returns, cov_matrix, n)
    print(f"  Equal-weight (1/{n}): Sharpe = {ew_sharpe:.4f}")

    # Classical mean-variance
    cmv_x, cmv_sharpe = run_classical_meanvar(mean_returns, cov_matrix, k)
    cmv_allocation = qubo.decode_bitstring(cmv_x) if cmv_x is not None else {}
    print(f"  Classical MV (K={k}): Sharpe = {cmv_sharpe:.4f}  "
          f"assets: {', '.join(cmv_allocation.keys())}")

    # Random
    random_stats = run_random_portfolios(mean_returns, cov_matrix, k)
    print(f"  Random (1000 samples): mean Sharpe = {random_stats['mean_sharpe']:.4f} "
          f"± {random_stats['std_sharpe']:.4f}")

    # ── Summary ──
    ew_annual = annualize_sharpe(ew_sharpe)
    cmv_annual = annualize_sharpe(cmv_sharpe)
    rand_annual = annualize_sharpe(random_stats['mean_sharpe'])

    print(f"\n{'='*70}")
    print(f"  SUMMARY  (annualized Sharpe, √{ANNUALIZE_HOURS} trading hours/yr)")
    print(f"{'='*70}")
    print(f"  {'Method':<25} {'Hourly':>10} {'Annual':>10} {'Assets':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'Equal Weight (1/N)':<25} {ew_sharpe:>10.4f} {ew_annual:>10.2f} {n:>8}")
    print(f"  {'Random K-of-N (mean)':<25} {random_stats['mean_sharpe']:>10.4f} {rand_annual:>10.2f} {k:>8}")
    print(f"  {'Classical MV (K-of-N)':<25} {cmv_sharpe:>10.4f} {cmv_annual:>10.2f} {k:>8}")
    print(f"  {'QAOA (K-of-N)':<25} {qaoa_portfolio['sharpe']:>10.4f} {qaoa_portfolio['sharpe_annual']:>10.2f} {qaoa_portfolio['n_assets']:>8}")
    if bf_portfolio:
        print(f"  {'Brute Force (optimal)':<25} {bf_portfolio['sharpe']:>10.4f} {bf_portfolio['sharpe_annual']:>10.2f} {bf_portfolio['n_assets']:>8}")
    print(f"{'='*70}")

    # ── Save results ──
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "assets": valid_assets,
            "n_assets": n,
            "cardinality_k": k,
            "risk_aversion": args.risk_aversion,
            "p_layers": args.p_layers,
            "mixer": args.mixer,
            "restarts": args.restarts,
            "maxiter": args.maxiter,
            "period": args.period,
        },
        "qaoa": {
            "bitstring": qaoa_result["best_bitstring"],
            "probability": qaoa_result["best_probability"],
            "allocation": {k: float(v) for k, v in qaoa_allocation.items()},
            "portfolio": qaoa_portfolio,
            "best_energy": qaoa_result["best_energy"],
            "optimization_time_s": round(qaoa_time, 1),
            "top_bitstrings": qaoa_result["top_bitstrings"][:5],
        },
        "baselines": {
            "equal_weight_sharpe": round(ew_sharpe, 6),
            "equal_weight_sharpe_annual": round(ew_annual, 4),
            "classical_mv_sharpe": round(cmv_sharpe, 6),
            "classical_mv_sharpe_annual": round(cmv_annual, 4),
            "classical_mv_assets": list(cmv_allocation.keys()),
            "random": random_stats,
            "random_sharpe_annual": round(rand_annual, 4),
        },
        "brute_force": {
            "sharpe": bf_portfolio["sharpe"] if bf_portfolio else None,
            "sharpe_annual": bf_portfolio["sharpe_annual"] if bf_portfolio else None,
            "assets": list(qubo.decode_bitstring(bf_x).keys()) if bf_x is not None else None,
            "top_portfolios": bf_top[:5] if bf_top else None,
        },
        "covariance_info": {
            "condition_number": round(float(cond), 1),
            "mean_returns_raw": {a: round(float(mean_returns[i]), 8) for i, a in enumerate(valid_assets)},
        },
    }

    out_file = args.json_out or str(RESULTS_DIR / "phase2d_qaoa_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
