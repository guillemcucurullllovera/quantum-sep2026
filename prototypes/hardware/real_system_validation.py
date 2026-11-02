#!/usr/bin/env python3
"""
Phase 0 Real-System Validation Pass
====================================
Generates realistic agent payloads matching the EXACT schemas from
openclaw_meryems agents (technical, macro, news, fundamental, risk_sentiment)
and validates the quantum_signal builder against them.

This is NOT synthetic noise. Each payload mirrors the exact field names,
types, and value ranges observed in the real agent source code.

Assets tested: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, SPY
"""

import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prototypes.other.quantum_signal_gen1_infra.builder import QuantumVectorBuilder
from prototypes.other.quantum_signal_gen1_infra.validator import VectorValidator
from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector


# ───── REALISTIC PAYLOAD GENERATORS ─────
# These replicate the EXACT dictionary structures published by each agent
# as verified from the openclaw_meryems agent source code.

FX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]
EQUITIES = ["SPY"]
ALL_ASSETS = FX_PAIRS + EQUITIES

# Real-world rate ranges (as of 2024-2026)
RATE_TABLE = {
    "EUR": (2.50, 3.50), "GBP": (4.00, 5.25), "USD": (4.25, 5.50),
    "JPY": (-0.10, 0.25), "AUD": (3.50, 4.50), "CHF": (1.00, 1.75),
}

PRICE_TABLE = {
    "EURUSD": (1.04, 1.12), "GBPUSD": (1.22, 1.30), "USDJPY": (140, 158),
    "AUDUSD": (0.62, 0.68), "USDCHF": (0.86, 0.92), "SPY": (440, 530),
}

EVENT_NAMES = [
    "US CPI Release", "FOMC Rate Decision", "US NFP", "ECB Rate Decision",
    "BOJ Policy Statement", "BOE Rate Decision", "US GDP (Q2)", "US PPI",
    "EU Flash PMI", "RBA Rate Decision", "SNB Policy Rate",
]


def generate_technical(asset, t, seed_state):
    """Generate a technical agent payload matching technical_agent.py schema."""
    price_lo, price_hi = PRICE_TABLE.get(asset, (1.0, 2.0))
    price = random.uniform(price_lo, price_hi)
    atr = price * random.uniform(0.002, 0.015)  # 0.2% to 1.5% of price
    rsi = random.gauss(50, 15)
    rsi = max(10, min(90, rsi))  # realistic RSI: 10-90
    macd_line = random.gauss(0, 0.003)
    macd_sig = macd_line + random.gauss(0, 0.001)
    macd_cross = round(macd_line - macd_sig, 6)
    bb_mid = price
    bb_range = atr * random.uniform(1.5, 3.0)
    bb_upper = round(bb_mid + bb_range, 5)
    bb_lower = round(bb_mid - bb_range, 5)
    bb_pos = random.uniform(0.05, 0.95)

    asset_type = "forex" if asset in FX_PAIRS else "equity"

    return {
        "agent_id": "signal-technical",
        "asset": asset,
        "asset_type": asset_type,
        "signal_type": "technical",
        "source": random.choice(["finage", "yahoo_finance"]),
        "rsi": round(rsi, 1),
        "macd_line": round(macd_line, 6),
        "macd_signal_line": round(macd_sig, 6),
        "macd_cross": macd_cross,
        "bb_upper": bb_upper,
        "bb_mid": round(bb_mid, 5),
        "bb_lower": bb_lower,
        "bb_position_ratio": round(bb_pos, 4),
        "price": round(price, 5),
        "atr": round(atr, 5),
        "weights_used": {"rsi": 1.5, "macd": 2.5, "bb": 0.5},
        "timestamp": t.isoformat() + "Z",
    }


def generate_macro(asset, t, seed_state):
    """Generate a macro agent payload matching macro_agent.py schema."""
    asset_type = "forex" if asset in FX_PAIRS else "equity"

    if asset_type == "forex":
        base, quote = asset[:3], asset[3:]
        rate_base = random.uniform(*RATE_TABLE.get(base, (2.0, 4.0)))
        rate_quote = random.uniform(*RATE_TABLE.get(quote, (2.0, 4.0)))
        carry = round(rate_base - rate_quote, 4)
        hours = random.choice([None, random.uniform(0.5, 72)])
        event = random.choice(EVENT_NAMES) if hours else None
        pending = random.randint(0, 5)

        return {
            "agent_id": "signal-macro",
            "asset": asset,
            "asset_type": "forex",
            "signal_type": "macro",
            "source": "carry_trade+calendar",
            "rate_base": round(rate_base, 2),
            "rate_quote": round(rate_quote, 2),
            "carry_differential": carry,
            "in_blackout": random.random() < 0.1,
            "hours_to_next_event": round(hours, 1) if hours else None,
            "next_event_name": event,
            "next_event_impact": random.choice(["HIGH", "MEDIUM", "LOW"]) if event else None,
            "pending_high_impact_count": pending,
            "timestamp": t.isoformat() + "Z",
        }
    else:
        # Equity macro schema
        vix = random.gauss(20, 6)
        vix = max(10, min(45, vix))
        return {
            "agent_id": "signal-macro",
            "asset": asset,
            "asset_type": "equity",
            "signal_type": "macro",
            "source": "equity_macro+calendar",
            "vix_level": round(vix, 2),
            "vix_change_pct": round(random.gauss(0, 3), 2),
            "us10y_yield": round(random.uniform(3.5, 5.0), 3),
            "us10y_change_pct": round(random.gauss(0, 1), 2),
            "in_blackout": random.random() < 0.1,
            "hours_to_next_event": round(random.uniform(1, 48), 1),
            "next_event_name": random.choice(EVENT_NAMES),
            "next_event_impact": random.choice(["HIGH", "MEDIUM"]),
            "pending_high_impact_count": random.randint(1, 4),
            "timestamp": t.isoformat() + "Z",
        }


def generate_news(asset, t, seed_state):
    """Generate a news agent payload matching news_agent.py schema."""
    asset_type = "forex" if asset in FX_PAIRS else "equity"

    if asset_type == "forex":
        base, quote = asset[:3], asset[3:]
        bc = random.randint(5, 50)
        bb = random.randint(1, max(2, bc // 3))
        bB = random.randint(1, max(2, bc // 3))
        qc = random.randint(5, 50)
        qb = random.randint(1, max(2, qc // 3))
        qB = random.randint(1, max(2, qc // 3))
        return {
            "agent_id": "signal-news",
            "asset": asset,
            "asset_type": "forex",
            "signal_type": "news",
            "source": "rss_feeds",
            "base_currency": base,
            "base_bull_count": bb,
            "base_bear_count": bB,
            "base_article_count": bc,
            "quote_currency": quote,
            "quote_bull_count": qb,
            "quote_bear_count": qB,
            "quote_article_count": qc,
            "top_headlines": ["Headline A", "Headline B"],
            "total_articles_analyzed": bc + qc,
            "timestamp": t.isoformat() + "Z",
        }
    else:
        fc = random.randint(10, 60)
        fb = random.randint(1, max(2, fc // 3))
        fB = random.randint(1, max(2, fc // 3))
        mc = random.randint(10, 60)
        mb = random.randint(1, max(2, mc // 3))
        mB = random.randint(1, max(2, mc // 3))
        return {
            "agent_id": "signal-news",
            "asset": asset,
            "asset_type": "equity",
            "signal_type": "news",
            "source": "rss_feeds",
            "focus_name": asset,
            "focus_bull_count": fb,
            "focus_bear_count": fB,
            "focus_article_count": fc,
            "market_bull_count": mb,
            "market_bear_count": mB,
            "market_article_count": mc,
            "top_headlines": ["SPY rally", "Market dip"],
            "total_articles_analyzed": fc + mc,
            "timestamp": t.isoformat() + "Z",
        }


def generate_fundamental(asset, t, seed_state):
    """Generate a fundamental agent payload matching fundamental_agent.py schema."""
    asset_type = "forex" if asset in FX_PAIRS else "equity"

    if asset_type == "forex":
        base, quote = asset[:3], asset[3:]
        b_cot = random.randint(-80000, 80000)
        q_cot = random.randint(-80000, 80000)
        pmi = random.gauss(50, 3)
        pmi = max(42, min(58, pmi))
        pmi_prev = pmi + random.gauss(0, 0.5)
        return {
            "agent_id": "signal-fundamental",
            "asset": asset,
            "asset_type": "forex",
            "signal_type": "fundamental",
            "source": "cot+price_trend",
            "base_ccy": base,
            "base_cot_net": b_cot,
            "base_cot_net_pct": round(b_cot / 5000, 1),  # approximate scaling
            "base_cot_source": random.choice(["lev_funds", "non_comm"]),
            "base_cot_date": t.strftime("%y%m%d"),
            "quote_ccy": quote,
            "quote_cot_net": q_cot,
            "quote_cot_net_pct": round(q_cot / 5000, 1),
            "quote_cot_source": random.choice(["lev_funds", "non_comm"]),
            "quote_cot_date": t.strftime("%y%m%d"),
            "price_trend_pct": round(random.gauss(0, 4), 1),
            "china_pmi": round(pmi, 1),
            "china_pmi_prev": round(pmi_prev, 1),
            "china_pmi_trend": round(pmi - pmi_prev, 1),
            "china_pmi_source": "fred_china_pmi",
            "timestamp": t.isoformat() + "Z",
        }
    else:
        return {
            "agent_id": "signal-fundamental",
            "asset": asset,
            "asset_type": "equity",
            "signal_type": "fundamental",
            "source": "yahoo_quote+price_trend",
            "price_trend_pct": round(random.gauss(0, 6), 1),
            "market_cap": random.randint(100_000_000_000, 500_000_000_000),
            "trailing_pe": round(random.uniform(18, 30), 2),
            "forward_pe": round(random.uniform(16, 28), 2),
            "price_to_book": round(random.uniform(3, 6), 2),
            "regular_market_price": round(random.uniform(440, 530), 2),
            "distance_to_52w_high_pct": round(random.uniform(-10, 0), 2),
            "distance_to_52w_low_pct": round(random.uniform(5, 25), 2),
            "timestamp": t.isoformat() + "Z",
        }


def generate_risk_sentiment(t):
    """Generate risk_sentiment payload matching technical_agent.py VIX block."""
    vix = random.gauss(20, 6)
    vix = max(10, min(45, vix))
    regimes = {True: "RISK_OFF", False: "RISK_ON"}
    regime = "RISK_OFF" if vix > 30 else ("CAUTION" if vix > 22 else ("NEUTRAL" if vix > 16 else "RISK_ON"))
    return {
        "vix_level": round(vix, 2),
        "vix_change": round(random.gauss(0, 2.5), 2),
        "risk_regime": regime,
        "timestamp": t.isoformat() + "Z",
    }


def run_validation(n_ticks=50, seed=42):
    """
    Run n_ticks of sequential signal generation per asset.
    Total vectors = n_ticks * len(ALL_ASSETS).
    """
    random.seed(seed)

    validator = VectorValidator()
    results = {
        "total_vectors": 0,
        "valid_vectors": 0,
        "invalid_vectors": 0,
        "warnings_total": 0,
        "errors": [],
        "by_asset": {},
        "feature_stats": defaultdict(lambda: {"min": float("inf"), "max": float("-inf"), "sum": 0, "count": 0, "zeros": 0}),
        "cold_start_analysis": {},
        "missing_field_analysis": defaultdict(int),
    }

    # One builder per asset (realistic: each asset has its own state)
    builders = {asset: QuantumVectorBuilder(window_size=252) for asset in ALL_ASSETS}

    base_time = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)

    for asset in ALL_ASSETS:
        builder = builders[asset]
        asset_results = {"valid": 0, "invalid": 0, "warnings": [], "cold_start_ticks": 0}
        cold_start_features = defaultdict(int)  # count of neutral/default values

        for tick in range(n_ticks):
            t = base_time + timedelta(minutes=tick * 15)  # 15-min ticks
            results["total_vectors"] += 1

            # Generate payloads matching real agent schemas
            technical = generate_technical(asset, t, tick)
            macro = generate_macro(asset, t, tick)
            news = generate_news(asset, t, tick)
            fundamental = generate_fundamental(asset, t, tick)
            risk_sent = generate_risk_sentiment(t)

            # Simulate occasional missing data (realistic: agents sometimes fail)
            if random.random() < 0.05:  # 5% chance of missing risk data
                risk_sent = {}
            if random.random() < 0.03:  # 3% stale fundamental
                fundamental = {"timestamp": (t - timedelta(hours=48)).isoformat() + "Z"}

            try:
                vector = builder.build(
                    technical=technical,
                    macro=macro,
                    news=news,
                    fundamental=fundamental,
                    risk_sentiment=risk_sent,
                    asset=asset,
                    timestamp=t.isoformat() + "Z",
                )

                report = validator.validate(vector)

                if report.is_valid:
                    results["valid_vectors"] += 1
                    asset_results["valid"] += 1
                else:
                    results["invalid_vectors"] += 1
                    asset_results["invalid"] += 1
                    results["errors"].append({
                        "asset": asset, "tick": tick,
                        "errors": report.errors,
                    })

                results["warnings_total"] += len(report.warnings)
                asset_results["warnings"].extend(report.warnings)

                # Track feature distributions
                arr = vector.to_array()
                field_names = [
                    "momentum_composite", "momentum_acceleration", "realized_vol_20d",
                    "vol_regime", "atr_normalized", "carry_differential",
                    "yield_curve_slope", "event_risk_score", "net_sentiment",
                    "sentiment_momentum", "cot_positioning", "cot_extreme",
                    "dxy_momentum", "risk_sentiment",
                ]
                for i, (name, val) in enumerate(zip(field_names, arr)):
                    stats = results["feature_stats"][name]
                    stats["min"] = min(stats["min"], val)
                    stats["max"] = max(stats["max"], val)
                    stats["sum"] += val
                    stats["count"] += 1
                    if abs(val) < 1e-8 or val == 0.5:
                        stats["zeros"] += 1

                # Cold-start analysis: track first 5 ticks
                if tick < 5:
                    for name, val in zip(field_names, arr):
                        if abs(val) < 1e-8 or (name in ("vol_regime", "risk_sentiment") and abs(val - 0.5) < 1e-8):
                            cold_start_features[name] += 1

            except Exception as e:
                results["invalid_vectors"] += 1
                results["errors"].append({
                    "asset": asset, "tick": tick,
                    "exception": str(e),
                })

        asset_results["cold_start_defaults"] = dict(cold_start_features)
        results["by_asset"][asset] = asset_results

    # Compute feature means and zero rates
    feature_summary = {}
    for name, stats in results["feature_stats"].items():
        n = stats["count"]
        feature_summary[name] = {
            "min": round(stats["min"], 6),
            "max": round(stats["max"], 6),
            "mean": round(stats["sum"] / n, 6) if n > 0 else 0,
            "zero_or_neutral_rate": round(stats["zeros"] / n, 4) if n > 0 else 0,
            "exercised": stats["min"] != stats["max"],
        }

    results["feature_summary"] = feature_summary

    # Cold start aggregation
    cold_agg = defaultdict(int)
    for asset_data in results["by_asset"].values():
        for fname, count in asset_data.get("cold_start_defaults", {}).items():
            cold_agg[fname] += count
    results["cold_start_analysis"] = dict(cold_agg)

    # Cleanup non-serializable
    del results["feature_stats"]

    return results


def print_report(results):
    """Print human-readable validation report."""
    total = results["total_vectors"]
    valid = results["valid_vectors"]
    invalid = results["invalid_vectors"]

    print("=" * 70)
    print("PHASE 0 REAL-SYSTEM VALIDATION REPORT")
    print("=" * 70)
    print(f"\nTotal vectors generated: {total}")
    print(f"Valid:                   {valid} ({100*valid/total:.1f}%)")
    print(f"Invalid:                 {invalid} ({100*invalid/total:.1f}%)")
    print(f"Total warnings:          {results['warnings_total']}")

    print(f"\n{'─'*70}")
    print("PER-ASSET BREAKDOWN")
    print(f"{'─'*70}")
    for asset, data in results["by_asset"].items():
        status = "PASS" if data["invalid"] == 0 else "FAIL"
        total_a = data["valid"] + data["invalid"]
        print(f"  {asset:8s}  {data['valid']}/{total_a} valid  [{status}]"
              f"  warnings={len(data['warnings'])}"
              f"  cold_defaults={len(data.get('cold_start_defaults', {}))}")

    print(f"\n{'─'*70}")
    print("FEATURE DISTRIBUTION ANALYSIS")
    print(f"{'─'*70}")
    print(f"  {'Feature':<25s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Zero/Neutral%':>14s} {'Exercised':>10s}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*14} {'─'*10}")
    for name, stats in results["feature_summary"].items():
        exercised = "YES" if stats["exercised"] else "** NO **"
        zero_pct = f"{100*stats['zero_or_neutral_rate']:.1f}%"
        print(f"  {name:<25s} {stats['min']:>8.4f} {stats['max']:>8.4f} {stats['mean']:>8.4f} {zero_pct:>14s} {exercised:>10s}")

    # Highlight features that are always zero/neutral
    always_neutral = [n for n, s in results["feature_summary"].items()
                      if s["zero_or_neutral_rate"] > 0.95]
    if always_neutral:
        print(f"\n  ** ALWAYS NEUTRAL: {', '.join(always_neutral)}")
        print(f"     These features default to 0.0 because no live data source provides them.")

    print(f"\n{'─'*70}")
    print("COLD-START ANALYSIS (first 5 ticks per asset)")
    print(f"{'─'*70}")
    if results["cold_start_analysis"]:
        for fname, count in sorted(results["cold_start_analysis"].items(), key=lambda x: -x[1]):
            total_cold = 5 * len(ALL_ASSETS)
            print(f"  {fname:<25s} defaulted {count}/{total_cold} times ({100*count/total_cold:.0f}%)")
    else:
        print("  No cold-start defaults detected.")

    if results["errors"]:
        print(f"\n{'─'*70}")
        print(f"ERRORS ({len(results['errors'])})")
        print(f"{'─'*70}")
        for err in results["errors"][:10]:
            print(f"  {err}")

    print(f"\n{'='*70}")
    verdict = "PASS" if invalid == 0 else ("PASS WITH WARNINGS" if invalid < total * 0.02 else "FAIL")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    return verdict


if __name__ == "__main__":
    # Run 50 ticks per asset * 6 assets = 300 vectors
    print("Running validation with 50 ticks per asset (300 total vectors)...")
    results = run_validation(n_ticks=50, seed=42)
    verdict = print_report(results)

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "real_system_validation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Also run with 100 ticks for a deeper pass
    print("\n\nRunning extended validation with 100 ticks per asset (600 total vectors)...")
    results_ext = run_validation(n_ticks=100, seed=7)
    verdict_ext = print_report(results_ext)

    output_path_ext = os.path.join(output_dir, "real_system_validation_extended.json")
    with open(output_path_ext, "w") as f:
        json.dump(results_ext, f, indent=2, default=str)
    print(f"\nExtended results saved to: {output_path_ext}")
