#!/usr/bin/env python3 -u
"""
QAOA Walk-Forward Backtester
==============================
Rolling-window backtest comparing QAOA portfolio allocation vs baselines.

At each rebalance point:
  1. Compute trailing covariance from last lookback_days
  2. Compute expected returns from trailing momentum
  3. Run QAOA optimizer → allocation weights
  4. Run classical MV optimizer → allocation weights
  5. Equal-weight baseline
  6. Compute next-period returns for all strategies
  7. Track cumulative returns, Sharpe, drawdown

Usage:
  python scripts/qaoa_backtest.py                                 # Default settings
  python scripts/qaoa_backtest.py --assets GBPUSD USDJPY SPY NVDA EURUSD
  python scripts/qaoa_backtest.py --rebalance-hours 24            # Daily rebalance
  python scripts/qaoa_backtest.py --lookback-days 20 --p-layers 3
  python scripts/qaoa_backtest.py --cardinality 3                 # Hold 3 assets
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from itertools import combinations
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# HISTORICAL PROTOTYPE — preserved from the Gen1 research repository,
# phase0/scripts/qaoa_backtest.py. Import path updated to this repo's
# layout; algorithmic logic below is UNCHANGED from the original.
#
# NOT independently runnable end-to-end: `fetch_yahoo_prices` (live Yahoo
# Finance fetch) and `adaptive_qubo_scaling` were not carried into
# algorithms/qaoa_qubo/gen1/qaoa_portfolio.py during port (network
# data-fetch layer dropped). Both are called only inside function bodies
# below (run_backtest / the CLI path), not at import time, so this module
# still imports cleanly — it just can't complete a live data-driven
# backtest without those two functions restored. See
# phase2d_qaoa_optimize.py's header for the same gap and rationale.
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


def ledoit_wolf_shrink(S, n_samples):
    """
    Ledoit-Wolf shrinkage estimator for covariance matrix.
    Shrinks sample covariance toward scaled identity (constant correlation).
    Reduces extreme eigenvalues caused by estimation noise.
    """
    n = S.shape[0]
    if n_samples < 2:
        return S

    # Target: scaled identity (average variance on diagonal)
    mu_target = np.trace(S) / n
    F = mu_target * np.eye(n)

    # Compute optimal shrinkage intensity (simplified Ledoit-Wolf)
    # delta = Frobenius distance
    delta = S - F
    delta_sq_sum = np.sum(delta ** 2)

    # Shrinkage intensity: higher when n_samples is small relative to n
    # Simple formula: alpha ~ n / n_samples (bounded [0, 1])
    alpha = min(1.0, max(0.0, n / (n_samples * 2)))

    return (1 - alpha) * S + alpha * F


def momentum_returns(all_returns, lookback):
    """
    Compute momentum-based expected returns instead of raw mean.
    Uses exponentially-weighted mean with recent returns weighted more.
    More robust than simple mean for assets with trend/momentum.
    """
    window = all_returns[-lookback:]
    n_rows = len(window)
    if n_rows < 10:
        return None

    # Exponential weights: recent returns matter more
    halflife = max(n_rows // 3, 5)
    decay = np.log(2) / halflife
    weights = np.exp(-decay * np.arange(n_rows)[::-1])
    weights /= weights.sum()

    # Weighted mean returns
    mu = (window * weights[:, None]).sum(axis=0)
    return mu


def compute_rolling_cov_and_returns(window_returns, full_returns, lookback,
                                     use_shrinkage=True, use_momentum=True):
    """
    Compute covariance and expected returns from a lookback window.

    Args:
        window_returns: pre-sliced lookback window for covariance
        full_returns: full return matrix up to current index (for momentum)
        lookback: number of bars to look back
        use_shrinkage: apply Ledoit-Wolf shrinkage to covariance
        use_momentum: use exponential momentum estimator for returns
    """
    if len(window_returns) < 10:
        return None, None

    # Covariance with optional shrinkage
    cov = np.cov(window_returns, rowvar=False)
    if use_shrinkage:
        cov = ledoit_wolf_shrink(cov, len(window_returns))

    # Returns: momentum-based or raw mean
    if use_momentum:
        mu = momentum_returns(full_returns, lookback)
        if mu is None:
            mu = np.mean(window_returns, axis=0)
    else:
        mu = np.mean(window_returns, axis=0)

    return cov, mu


def _score_portfolio(indices, mu, cov, k, min_variance):
    """Score a K-asset equal-weight portfolio. Lower=better for min-var, higher=better for Sharpe."""
    n = len(mu)
    w = np.zeros(n)
    for i in indices:
        w[i] = 1.0 / k
    risk = w @ cov @ w
    if min_variance:
        return -risk  # negate so higher = better universally
    ret = mu @ w
    return ret / max(np.sqrt(max(risk, 1e-20)), 1e-15)


def classical_mv_select(mu, cov, k, min_variance=False):
    """
    Select K assets using classical exhaustive search (exact optimum).
    This is the oracle baseline — no heuristic can beat it.
    """
    n = len(mu)
    best_score = -1e18
    best_selection = None

    for combo in combinations(range(n), k):
        score = _score_portfolio(combo, mu, cov, k, min_variance)
        if score > best_score:
            best_score = score
            best_selection = list(combo)

    x = np.zeros(n)
    if best_selection:
        for i in best_selection:
            x[i] = 1.0
    return x


def greedy_forward_select(mu, cov, k, min_variance=False):
    """
    Greedy forward selection: start empty, add the asset that improves
    the portfolio score the most at each step. O(N*K) — fast but myopic.
    This is what a practitioner would use when N is too large for brute force.
    """
    n = len(mu)
    selected = []
    remaining = list(range(n))

    for _ in range(k):
        best_score = -1e18
        best_asset = remaining[0]
        for a in remaining:
            trial = selected + [a]
            score = _score_portfolio(trial, mu, cov, len(trial), min_variance)
            if score > best_score:
                best_score = score
                best_asset = a
        selected.append(best_asset)
        remaining.remove(best_asset)

    x = np.zeros(n)
    for i in selected:
        x[i] = 1.0
    return x


def random_best_of_n_select(mu, cov, k, min_variance=False, n_samples=50):
    """
    Random sampling baseline: generate n_samples random K-subsets,
    return the best one. Simple Monte Carlo approach.
    """
    n = len(mu)
    best_score = -1e18
    best_x = None

    for _ in range(n_samples):
        indices = tuple(sorted(np.random.choice(n, size=k, replace=False)))
        score = _score_portfolio(indices, mu, cov, k, min_variance)
        if score > best_score:
            best_score = score
            best_x = np.zeros(n)
            for i in indices:
                best_x[i] = 1.0

    if best_x is None:
        best_x = np.zeros(n)
        best_x[:k] = 1.0
    return best_x


def simulated_annealing_select(mu, cov, k, min_variance=False, n_steps=200):
    """
    Simulated annealing: start from random K-subset, swap one asset at a time,
    accept worse moves with decreasing probability. Good heuristic baseline.
    """
    n = len(mu)
    # Initial random solution
    current = list(sorted(np.random.choice(n, size=k, replace=False)))
    current_score = _score_portfolio(current, mu, cov, k, min_variance)
    best = list(current)
    best_score = current_score

    for step in range(n_steps):
        temp = 1.0 * (1 - step / n_steps)  # linear cooling
        # Swap: remove one selected asset, add one unselected
        out_set = [i for i in range(n) if i not in current]
        if not out_set:
            break
        remove_idx = np.random.randint(k)
        add_asset = out_set[np.random.randint(len(out_set))]

        trial = list(current)
        trial[remove_idx] = add_asset
        trial_score = _score_portfolio(trial, mu, cov, k, min_variance)

        delta = trial_score - current_score
        if delta > 0 or (temp > 0 and np.random.random() < np.exp(delta / max(temp, 1e-10))):
            current = trial
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)

    x = np.zeros(n)
    for i in best:
        x[i] = 1.0
    return x


def qaoa_select(mu_raw, cov_raw, k, p_layers=3, restarts=5, maxiter=200,
                risk_aversion=None, min_variance=False, q_values_override=None):
    """
    Run QAOA to select K assets with adaptive scaling.

    Three modes:
      1. min_variance=True: QAOA minimizes portfolio variance only (q=1.0, mu=0)
         Most robust mode — ignores noisy return estimates.
      2. risk_aversion=float: fixed q value
      3. risk_aversion=None: sweeps [0.3, 0.5, 0.7, 0.9] picks best

    Args:
        mu_raw: (N,) raw mean returns (NOT pre-scaled)
        cov_raw: (N,N) raw covariance (NOT pre-scaled)
        k: cardinality
        risk_aversion: Fixed q value. If None, sweeps multiple values.
        min_variance: If True, ignore returns entirely (pure risk minimization).
    """
    n = len(mu_raw)

    if min_variance:
        # Pure risk minimization: zero out returns, q=1.0
        mu_for_qubo = np.zeros(n)
        cov_for_qubo = cov_raw / max(np.max(np.abs(cov_raw)), 1e-20)  # normalize
        q_values = [1.0]
    else:
        # Adaptive scaling: balance return vs risk terms in the QUBO
        mu_for_qubo, cov_for_qubo = adaptive_qubo_scaling(mu_raw, cov_raw, k,
                                                           target_return_frac=0.4)
        if risk_aversion is not None:
            q_values = [risk_aversion]
        elif q_values_override is not None:
            q_values = list(q_values_override)
        else:
            q_values = [0.3, 0.5, 0.7, 0.9]

    best_x = None
    best_score = float('inf') if min_variance else -999

    for q in q_values:
        qubo = PortfolioQUBO(
            expected_returns=mu_for_qubo,
            cov_matrix=cov_for_qubo,
            risk_aversion=q,
            cardinality_k=k,
            penalty_lambda=0.0,
        )
        J, h, offset = qubo.build_ising()

        circuit = QAOACircuit(
            n_assets=n, J=J, h=h,
            p_layers=p_layers,
            mixer="xy",
            cardinality_k=k,
        )

        optimizer = QAOAOptimizer(
            circuit, qubo,
            maxiter=maxiter,
            n_restarts=restarts,
        )

        result = optimizer.optimize(verbose=False)
        x = np.array([int(b) for b in result["best_bitstring"]], dtype=float)

        sel = np.where(x > 0.5)[0]
        if len(sel) == 0:
            continue
        w = np.zeros(n)
        w[sel] = 1.0 / len(sel)

        if min_variance:
            # Score by portfolio variance (lower = better)
            var = w @ cov_raw @ w
            if var < best_score:
                best_score = var
                best_x = x
        else:
            # Score by Sharpe on lookback data (higher = better)
            ret = mu_raw @ w
            risk = np.sqrt(max(w @ cov_raw @ w, 1e-20))
            sharpe = ret / max(risk, 1e-15)
            if sharpe > best_score:
                best_score = sharpe
                best_x = x

    if best_x is None:
        # Fallback: random K-subset (not all-N, which would contaminate comparison)
        best_x = np.zeros(n)
        chosen = np.random.choice(n, size=k, replace=False)
        best_x[chosen] = 1.0

    return best_x


def compute_period_return(returns_row, selection, k):
    """Compute portfolio return for one period given selection and equal weights."""
    w = selection / max(np.sum(selection), 1)
    return float(returns_row @ w)


def bootstrap_sharpe_ci(returns, periods_per_year, n_boot=1000, alpha=0.05, seed=7):
    """Bootstrap 95% CI on annualized Sharpe."""
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns)
    n = len(returns)
    if n < 5:
        return None, None
    sharpes = []
    sqrt_pyr = np.sqrt(periods_per_year)
    for _ in range(n_boot):
        sample = rng.choice(returns, size=n, replace=True)
        m = float(np.mean(sample))
        sd = float(np.std(sample))
        if sd < 1e-10:
            continue
        sharpes.append(m / sd * sqrt_pyr)
    if not sharpes:
        return None, None
    lo = float(np.percentile(sharpes, 100 * alpha / 2))
    hi = float(np.percentile(sharpes, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4)


def compute_metrics(returns_series, periods_per_year=252, bootstrap=False):
    """Compute Sharpe, max drawdown, Calmar from a return series.

    Args:
        returns_series: list of per-period LOG returns
        periods_per_year: how many rebalance periods fit in one year
            e.g. daily rebalance = 252, weekly (5-day) = 252/5 = 50.4
        bootstrap: if True, compute 95% bootstrap CI on Sharpe
    """
    returns = np.array(returns_series)
    if len(returns) == 0:
        return {"sharpe": 0, "max_dd": 0, "calmar": 0, "total_return": 0}

    mean_r = np.mean(returns)
    std_r = np.std(returns)
    sharpe = mean_r / max(std_r, 1e-10) * np.sqrt(periods_per_year)

    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))

    total_ret = float(cum[-1] - 1)
    calmar = total_ret / max(abs(max_dd), 1e-10) if max_dd < 0 else total_ret * 10

    out = {
        "sharpe": round(float(sharpe), 4),
        "max_dd": round(float(max_dd), 4),
        "calmar": round(float(calmar), 4),
        "total_return": round(total_ret, 6),
        "n_periods": len(returns),
        "mean_period_return": round(float(mean_r), 8),
    }
    if bootstrap:
        lo, hi = bootstrap_sharpe_ci(returns, periods_per_year)
        out["sharpe_ci_lo"] = lo
        out["sharpe_ci_hi"] = hi
    return out


def regime_split_metrics(strat_returns, ew_returns, periods_per_year):
    """Split periods into bull (EW>0) / bear (EW<0) / flat, compute Sharpe in each."""
    sr = np.asarray(strat_returns)
    er = np.asarray(ew_returns)
    # Use EW return to classify period regime
    bull_mask = er > 0.005
    bear_mask = er < -0.005
    flat_mask = ~bull_mask & ~bear_mask

    out = {}
    for label, mask in [("bull", bull_mask), ("bear", bear_mask), ("flat", flat_mask)]:
        if mask.sum() >= 3:
            sub = sr[mask]
            m = float(np.mean(sub))
            sd = float(np.std(sub))
            s = m / max(sd, 1e-10) * np.sqrt(periods_per_year)
            out[label] = {
                "sharpe": round(s, 4),
                "mean_return": round(m, 6),
                "n_periods": int(mask.sum()),
            }
        else:
            out[label] = {"sharpe": None, "n_periods": int(mask.sum())}
    return out


def main():
    parser = argparse.ArgumentParser(description="QAOA Walk-Forward Backtest")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--period", default="60d", help="Total price data period")
    parser.add_argument("--interval", default="1h", choices=["1h", "1d"],
                        help="Bar interval: 1h for hourly, 1d for daily (default: 1h)")
    parser.add_argument("--lookback-days", type=int, default=10,
                        help="Lookback for covariance estimation (default: 10 days)")
    parser.add_argument("--rebalance-hours", type=int, default=24,
                        help="Rebalance frequency in hours (default: 24)")
    parser.add_argument("--rebalance-days", type=int, default=None,
                        help="Rebalance frequency in days (for daily interval)")
    parser.add_argument("--cardinality", type=int, default=None,
                        help="K: number of assets to hold. Default: N//2")
    parser.add_argument("--p-layers", type=int, default=3,
                        help="QAOA circuit depth (default: 3)")
    parser.add_argument("--restarts", type=int, default=5,
                        help="QAOA optimizer restarts per rebalance (default: 5)")
    parser.add_argument("--maxiter", type=int, default=200,
                        help="Max optimizer iterations (default: 200)")
    parser.add_argument("--min-var", action="store_true",
                        help="Minimum variance mode: ignore returns, minimize risk only")
    parser.add_argument("--no-shrinkage", action="store_true",
                        help="Disable Ledoit-Wolf covariance shrinkage")
    parser.add_argument("--no-momentum", action="store_true",
                        help="Disable momentum-based return estimator (use raw mean)")
    parser.add_argument("--q-values", nargs="+", type=float, default=None,
                        help="Override default q-value sweep for sharpe-opt mode")
    parser.add_argument("--tx-cost-bps", type=float, default=0.0,
                        help="Transaction cost in basis points per unit of turnover (default: 0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--long-short", action="store_true",
                        help="Long-K + Short-K mode (net-zero exposure). "
                             "Picks K best to long and K worst to short.")
    parser.add_argument("--bootstrap-ci", action="store_true",
                        help="Compute 95%% bootstrap CI on Sharpe for all strategies")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    # Set seed early for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)

    print("=" * 60)
    print("QAOA WALK-FORWARD BACKTEST")
    print("=" * 60)

    # ── Determine interval and rebalance settings ──
    is_daily = (args.interval == "1d")
    if is_daily:
        rebalance_bars = args.rebalance_days if args.rebalance_days else 5  # weekly default
        lookback_bars = args.lookback_days  # 1 bar = 1 day
        # Each return covers rebalance_bars trading days → periods/year = 252/rebalance_bars
        periods_per_year = 252.0 / rebalance_bars
        bar_label = "daily"
    else:
        rebalance_bars = args.rebalance_hours
        lookback_bars = args.lookback_days * 24  # 1 bar = 1 hour
        # Each return covers rebalance_bars hours → periods/year = (252*6.5)/rebalance_bars
        trading_hours_per_year = 252 * 6.5  # ~1638 trading hours
        periods_per_year = trading_hours_per_year / rebalance_bars
        bar_label = "hourly"

    # ── Fetch price data ──
    print(f"\n1. Fetching price data ({args.period}, {args.interval} bars)...")
    price_data = {}
    for asset in args.assets:
        bars = fetch_yahoo_prices(asset, period=args.period, interval=args.interval)
        if bars and len(bars) >= 50:
            price_data[asset] = bars
            print(f"  {asset}: {len(bars)} bars")
        else:
            print(f"  {asset}: FAILED (got {len(bars) if bars else 0})")

    valid_assets = [a for a in args.assets if a in price_data]
    n = len(valid_assets)
    if n < 3:
        print("ERROR: Need at least 3 assets")
        sys.exit(1)

    k = args.cardinality or max(2, n // 2)
    k = min(k, n - 1)
    rebalance_label = f"{rebalance_bars}d" if is_daily else f"{rebalance_bars}h"
    print(f"\n  {n} assets, K={k}, rebalance every {rebalance_label} ({bar_label} bars)")

    # ── Align price series ──
    # Find common timestamps and compute return matrices
    print(f"\n2. Aligning price series...")

    # Use closes, align by index (hourly bars)
    min_len = min(len(price_data[a]) for a in valid_assets)
    # Use last min_len bars for each asset
    closes_matrix = np.column_stack([
        np.array([b["close"] for b in price_data[a][-min_len:]])
        for a in valid_assets
    ])

    # Compute log returns
    returns_matrix = np.diff(np.log(closes_matrix), axis=0)
    n_bars = returns_matrix.shape[0]
    print(f"  {n_bars} {bar_label} returns × {n} assets")

    # Market context: show overall period return per asset so we know if it's bull/bear
    total_log_ret = np.sum(returns_matrix, axis=0)
    avg_ret = np.mean(total_log_ret)
    print(f"\n  Market context over period:")
    for i, a in enumerate(valid_assets):
        print(f"    {a}: {total_log_ret[i]:+.2%}")
    mkt_label = "BULL" if avg_ret > 0.05 else ("BEAR" if avg_ret < -0.05 else "FLAT")
    print(f"  Average: {avg_ret:+.2%} ({mkt_label} market)")
    if mkt_label == "BEAR":
        print(f"  NOTE: Long-only K-of-N selection cannot profit in a bear market.")
        print(f"  Evaluate QAOA by *relative* alpha vs baselines, not absolute Sharpe.")

    # ── Walk-forward loop ──
    rebalance_every = rebalance_bars

    start_idx = lookback_bars
    n_rebalances = 0

    qaoa_returns = []
    cmv_returns = []
    greedy_returns = []
    random_returns = []
    sa_returns = []
    ew_returns = []

    qaoa_selections = []
    cmv_selections = []

    use_shrinkage = not args.no_shrinkage
    use_momentum = not args.no_momentum
    min_var = args.min_var
    long_short = args.long_short
    tx_cost_bps = args.tx_cost_bps

    mode_label = "MIN-VARIANCE" if min_var else "SHARPE-OPT"
    if long_short:
        mode_label = mode_label + " | LONG-SHORT"
    print(f"\n3. Running walk-forward (lookback={args.lookback_days}d, "
          f"rebalance={rebalance_label})...")
    print(f"   QAOA: p={args.p_layers}, {args.restarts} restarts, {args.maxiter} iter")
    print(f"   Mode: {mode_label}  Shrinkage: {use_shrinkage}  Momentum-mu: {use_momentum}")
    print(f"   Search space: C({n},{k}) = {math.comb(n, k)} portfolios")
    print(f"   Tx cost: {tx_cost_bps} bps/turnover  ·  Seed: {args.seed}  ·  Bootstrap CI: {args.bootstrap_ci}")
    if long_short:
        print(f"   LONG-SHORT mode: K={k} longs + K={k} shorts (net-zero)")

    # Track previous weights for turnover computation
    prev_qaoa_w = np.zeros(n)
    prev_cmv_w = np.zeros(n)
    prev_greedy_w = np.zeros(n)
    prev_random_w = np.zeros(n)
    prev_sa_w = np.zeros(n)
    prev_ew_w = np.zeros(n)

    def make_weights(x_select, k_eff, ls_invert=None):
        """Build weight vector. ls_invert: if provided, second selection treated as shorts."""
        w = np.zeros(n)
        sel = np.where(x_select > 0.5)[0]
        if len(sel) > 0:
            w[sel] = 1.0 / k_eff
        if ls_invert is not None:
            short_sel = np.where(ls_invert > 0.5)[0]
            if len(short_sel) > 0:
                w[short_sel] = -1.0 / k_eff
        return w

    def apply_tx_cost(period_ret, w_new, w_prev):
        """Subtract turnover-based cost from a period return."""
        if tx_cost_bps <= 0:
            return period_ret
        turnover = float(np.sum(np.abs(w_new - w_prev)))
        cost = tx_cost_bps * 1e-4 * turnover
        return period_ret - cost

    t0 = time.time()
    idx = start_idx

    while idx + rebalance_every <= n_bars:
        # Get lookback window + full history up to current point
        window_returns = returns_matrix[idx - lookback_bars:idx]
        full_history = returns_matrix[:idx]  # all returns up to now for momentum
        cov, mu_hist = compute_rolling_cov_and_returns(
            window_returns, full_history, lookback_bars,
            use_shrinkage=use_shrinkage,
            use_momentum=use_momentum,
        )

        if cov is None:
            idx += rebalance_every
            continue

        # Forward returns for this period
        fwd_returns = returns_matrix[idx:idx + rebalance_every]
        if len(fwd_returns) == 0:
            idx += rebalance_every
            continue

        # Period return = sum of hourly log returns
        period_returns = np.sum(fwd_returns, axis=0)

        # QAOA selection (longs)
        try:
            qaoa_x = qaoa_select(
                mu_hist, cov, k,
                p_layers=args.p_layers,
                restarts=args.restarts,
                maxiter=args.maxiter,
                min_variance=min_var,
                q_values_override=args.q_values,
            )
        except Exception as e:
            print(f"  QAOA error at idx={idx}: {e}")
            qaoa_x = np.zeros(n)
            chosen = np.random.choice(n, size=k, replace=False)
            qaoa_x[chosen] = 1.0

        # Classical + heuristic baselines (longs only — same mode)
        cmv_x = classical_mv_select(mu_hist, cov, k, min_variance=min_var)
        greedy_x = greedy_forward_select(mu_hist, cov, k, min_variance=min_var)
        random_x = random_best_of_n_select(mu_hist, cov, k, min_variance=min_var, n_samples=50)
        sa_x = simulated_annealing_select(mu_hist, cov, k, min_variance=min_var, n_steps=200)
        ew_x = np.ones(n)

        # ── LONG-SHORT mode: also pick K shorts by inverting μ for QAOA ──
        # (For min-var mode there's no notion of 'short' — long-short only makes
        #  sense in sharpe-opt where μ direction matters.)
        qaoa_short = cmv_short = greedy_short = random_short = sa_short = None
        if long_short and not min_var:
            try:
                qaoa_short = qaoa_select(
                    -mu_hist, cov, k,  # negate μ so QAOA picks the worst → shorts
                    p_layers=args.p_layers,
                    restarts=args.restarts,
                    maxiter=args.maxiter,
                    min_variance=False,
                    q_values_override=args.q_values,
                )
            except Exception:
                qaoa_short = np.zeros(n)
                chosen = np.random.choice(n, size=k, replace=False)
                qaoa_short[chosen] = 1.0
            cmv_short = classical_mv_select(-mu_hist, cov, k, min_variance=False)
            greedy_short = greedy_forward_select(-mu_hist, cov, k, min_variance=False)
            random_short = random_best_of_n_select(-mu_hist, cov, k, min_variance=False, n_samples=50)
            sa_short = simulated_annealing_select(-mu_hist, cov, k, min_variance=False, n_steps=200)

        # Build weight vectors
        if long_short and not min_var:
            qaoa_w   = make_weights(qaoa_x,   k, ls_invert=qaoa_short)
            cmv_w    = make_weights(cmv_x,    k, ls_invert=cmv_short)
            greedy_w = make_weights(greedy_x, k, ls_invert=greedy_short)
            random_w = make_weights(random_x, k, ls_invert=random_short)
            sa_w     = make_weights(sa_x,     k, ls_invert=sa_short)
            ew_w     = np.ones(n) / n  # EW stays long-only
        else:
            qaoa_w   = make_weights(qaoa_x,   k)
            cmv_w    = make_weights(cmv_x,    k)
            greedy_w = make_weights(greedy_x, k)
            random_w = make_weights(random_x, k)
            sa_w     = make_weights(sa_x,     k)
            ew_w     = np.ones(n) / n

        # Compute portfolio returns from weight vectors
        qaoa_ret_g   = float(period_returns @ qaoa_w)
        cmv_ret_g    = float(period_returns @ cmv_w)
        greedy_ret_g = float(period_returns @ greedy_w)
        random_ret_g = float(period_returns @ random_w)
        sa_ret_g     = float(period_returns @ sa_w)
        ew_ret_g     = float(period_returns @ ew_w)

        # Apply transaction costs based on turnover from previous period
        qaoa_ret   = apply_tx_cost(qaoa_ret_g,   qaoa_w,   prev_qaoa_w)
        cmv_ret    = apply_tx_cost(cmv_ret_g,    cmv_w,    prev_cmv_w)
        greedy_ret = apply_tx_cost(greedy_ret_g, greedy_w, prev_greedy_w)
        random_ret = apply_tx_cost(random_ret_g, random_w, prev_random_w)
        sa_ret     = apply_tx_cost(sa_ret_g,     sa_w,     prev_sa_w)
        ew_ret     = apply_tx_cost(ew_ret_g,     ew_w,     prev_ew_w)

        # Update previous weights
        prev_qaoa_w   = qaoa_w
        prev_cmv_w    = cmv_w
        prev_greedy_w = greedy_w
        prev_random_w = random_w
        prev_sa_w     = sa_w
        prev_ew_w     = ew_w

        qaoa_returns.append(qaoa_ret)
        cmv_returns.append(cmv_ret)
        greedy_returns.append(greedy_ret)
        random_returns.append(random_ret)
        sa_returns.append(sa_ret)
        ew_returns.append(ew_ret)

        qaoa_selected = [valid_assets[i] for i in range(n) if qaoa_x[i] > 0.5]
        cmv_selected = [valid_assets[i] for i in range(n) if cmv_x[i] > 0.5]
        qaoa_selections.append(qaoa_selected)
        cmv_selections.append(cmv_selected)

        n_rebalances += 1
        if n_rebalances % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Rebalance {n_rebalances}: QAOA={qaoa_ret:+.4f} CMV={cmv_ret:+.4f} "
                  f"EW={ew_ret:+.4f}  ({elapsed:.0f}s)")

        idx += rebalance_every

    total_time = time.time() - t0
    print(f"\n  Completed {n_rebalances} rebalance periods in {total_time:.1f}s")

    if n_rebalances < 3:
        print("ERROR: Not enough rebalance periods for meaningful comparison")
        sys.exit(1)

    # ── Compute metrics ──
    print(f"\n4. Results:")

    bs = args.bootstrap_ci
    qaoa_metrics = compute_metrics(qaoa_returns, periods_per_year, bootstrap=bs)
    cmv_metrics = compute_metrics(cmv_returns, periods_per_year, bootstrap=bs)
    greedy_metrics = compute_metrics(greedy_returns, periods_per_year, bootstrap=bs)
    random_metrics = compute_metrics(random_returns, periods_per_year, bootstrap=bs)
    sa_metrics = compute_metrics(sa_returns, periods_per_year, bootstrap=bs)
    ew_metrics = compute_metrics(ew_returns, periods_per_year, bootstrap=bs)

    # Regime splits — classify each period as bull/bear/flat using EW return as benchmark
    qaoa_regime = regime_split_metrics(qaoa_returns, ew_returns, periods_per_year)
    ew_regime = regime_split_metrics(ew_returns, ew_returns, periods_per_year)

    all_strats = [
        ("QAOA (K-of-N)", qaoa_metrics, qaoa_returns),
        ("Classical MV (exact)", cmv_metrics, cmv_returns),
        ("Greedy Forward", greedy_metrics, greedy_returns),
        ("Random Best-of-50", random_metrics, random_returns),
        ("Sim. Annealing", sa_metrics, sa_returns),
        ("Equal Weight (1/N)", ew_metrics, ew_returns),
    ]

    print(f"\n  Annualization: {periods_per_year:.1f} periods/year   "
          f"Tx cost: {tx_cost_bps} bps   Long-short: {long_short}   Seed: {args.seed}")
    if bs:
        print(f"\n  {'Strategy':<22} {'Sharpe':>8} {'95% CI':>17} {'Total Ret':>10} {'Max DD':>8}")
        print(f"  {'-'*22} {'-'*8} {'-'*17} {'-'*10} {'-'*8}")
        for name, metrics, _ in all_strats:
            ci_str = f"[{metrics.get('sharpe_ci_lo','?'):>+5.2f},{metrics.get('sharpe_ci_hi','?'):>+5.2f}]"
            print(f"  {name:<22} {metrics['sharpe']:>8.3f} {ci_str:>17} "
                  f"{metrics['total_return']:>+10.4f} {metrics['max_dd']:>8.4f}")
    else:
        print(f"\n  {'Strategy':<22} {'Sharpe':>8} {'Total Ret':>10} {'Max DD':>8} {'Calmar':>8}")
        print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
        for name, metrics, _ in all_strats:
            print(f"  {name:<22} {metrics['sharpe']:>8.3f} "
                  f"{metrics['total_return']:>+10.4f} {metrics['max_dd']:>8.4f} "
                  f"{metrics['calmar']:>8.3f}")

    # Regime splits
    print(f"\n  QAOA regime breakdown (periods classified by EW return sign):")
    for label in ["bull", "bear", "flat"]:
        rd = qaoa_regime.get(label, {})
        ew_rd = ew_regime.get(label, {})
        if rd.get("sharpe") is not None:
            print(f"    {label.upper():<5}  QAOA Sharpe={rd['sharpe']:+.3f}  "
                  f"EW Sharpe={ew_rd.get('sharpe', 0):+.3f}  ({rd['n_periods']} periods)")
        else:
            print(f"    {label.upper():<5}  insufficient data ({rd.get('n_periods', 0)} periods)")

    # Win rates: QAOA vs each baseline
    print(f"\n  QAOA win rates (per-period):")
    for name, _, rets in all_strats[1:]:  # skip QAOA itself
        wins = sum(1 for q, b in zip(qaoa_returns, rets) if q > b)
        print(f"    vs {name:<22}: {wins}/{n_rebalances} ({100*wins/n_rebalances:.0f}%)")

    # Most selected assets by QAOA
    from collections import Counter
    qaoa_asset_freq = Counter()
    for sel in qaoa_selections:
        qaoa_asset_freq.update(sel)
    print(f"\n  QAOA asset selection frequency:")
    for asset, count in qaoa_asset_freq.most_common():
        print(f"    {asset}: {count}/{n_rebalances} ({100*count/n_rebalances:.0f}%)")

    # ── Save results ──
    # Win rate vs each baseline for JSON
    win_rates = {}
    for name, _, rets in all_strats[1:]:
        wins = sum(1 for q, b in zip(qaoa_returns, rets) if q > b)
        win_rates[name] = round(wins / n_rebalances, 4)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "assets": valid_assets,
            "n_assets": n,
            "cardinality_k": k,
            "search_space": math.comb(n, k),
            "interval": args.interval,
            "lookback_days": args.lookback_days,
            "lookback_bars": lookback_bars,
            "rebalance_bars": rebalance_every,
            "rebalance_label": rebalance_label,
            "periods_per_year": round(periods_per_year, 2),
            "mode": mode_label,
            "use_shrinkage": use_shrinkage,
            "use_momentum": use_momentum,
            "p_layers": args.p_layers,
            "restarts": args.restarts,
            "maxiter": args.maxiter,
            "period": args.period,
            "n_rebalances": n_rebalances,
            "total_time_s": round(total_time, 1),
        },
        "metrics": {
            "qaoa": qaoa_metrics,
            "classical_mv_exact": cmv_metrics,
            "greedy_forward": greedy_metrics,
            "random_best_of_50": random_metrics,
            "simulated_annealing": sa_metrics,
            "equal_weight": ew_metrics,
        },
        "qaoa_win_rates": win_rates,
        "qaoa_regime_split": qaoa_regime,
        "ew_regime_split": ew_regime,
        "tx_cost_bps": tx_cost_bps,
        "long_short": long_short,
        "seed": args.seed,
        "period_returns": {
            "qaoa": qaoa_returns,
            "classical_mv_exact": cmv_returns,
            "greedy_forward": greedy_returns,
            "random_best_of_50": random_returns,
            "simulated_annealing": sa_returns,
            "equal_weight": ew_returns,
        },
        "qaoa_selections": qaoa_selections,
        "qaoa_asset_frequency": dict(qaoa_asset_freq.most_common()),
    }

    # Auto-generate unique filename from config to prevent overwrites
    if args.json_out:
        out_file = args.json_out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{n}asset_{mode_label.lower()}_{args.interval}_{rebalance_label}"
        out_file = str(RESULTS_DIR / f"qaoa_backtest_{tag}_{ts}.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_file}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
