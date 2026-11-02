#!/usr/bin/env python3
"""
Multi-Capture Pipeline
=======================
Runs multiple redis capture cycles with delays between them, then processes
all accumulated snapshots through the canonical builder with rolling state.

This proves that stateful features (momentum_acceleration, vol_regime,
sentiment_momentum, carry_differential, risk_sentiment, cot_extreme)
wake up after multiple capture ticks.

Usage:
  # Capture 5 cycles, 70s apart (one technical cycle = 60s), then validate
  python scripts/multi_capture_pipeline.py --cycles 5 --interval 70 --redis-port 6380

  # Process existing snapshots only (no new capture)
  python scripts/multi_capture_pipeline.py --skip-capture --enrich

  # Dry run with fixtures (no Redis needed)
  python scripts/multi_capture_pipeline.py --fixture-mode --cycles 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prototypes.other.quantum_signal_gen1_infra.builder import QuantumVectorBuilder
from prototypes.other.quantum_signal_gen1_infra.validator import VectorValidator
from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector

PHASE0_DIR = Path(__file__).parent.parent
SNAPSHOT_DIR = PHASE0_DIR / "snapshots"
RESULTS_DIR = PHASE0_DIR / "results"

FEATURE_NAMES = [
    "momentum_composite", "momentum_acceleration", "realized_vol_20d",
    "vol_regime", "atr_normalized", "carry_differential",
    "yield_curve_slope", "event_risk_score", "net_sentiment",
    "sentiment_momentum", "cot_positioning", "cot_extreme",
    "dxy_momentum", "risk_sentiment",
]

REDUCED_8Q_NAMES = [
    "momentum_composite", "realized_vol_20d", "vol_regime",
    "carry_differential", "event_risk_score", "net_sentiment",
    "cot_positioning", "risk_sentiment",
]

REDUCED_8Q_INDICES = [0, 2, 3, 5, 7, 8, 10, 13]


def run_capture(redis_port=6379, redis_host="localhost"):
    """Run one capture cycle via redis_capture.py"""
    cmd = [
        sys.executable, str(PHASE0_DIR / "scripts" / "redis_capture.py"),
        "--once",
        "--redis-host", redis_host,
        "--redis-port", str(redis_port),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Capture failed: {result.stderr[:200]}")
        return False
    # Count snapshots saved
    for line in result.stdout.splitlines():
        if "Saved" in line or "snapshots" in line.lower():
            print(f"  {line.strip()}")
    return True


def load_all_snapshots(snapshot_dir, asset_filter=None):
    """Load all snapshots sorted by time, grouped by asset."""
    snapshot_path = Path(snapshot_dir)
    if not snapshot_path.exists():
        return []

    snapshots = []
    for asset_dir in sorted(snapshot_path.iterdir()):
        if not asset_dir.is_dir() or asset_dir.name.startswith("_"):
            continue
        asset = asset_dir.name
        if asset_filter and asset != asset_filter:
            continue
        for json_file in sorted(asset_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                ts = data.get("captured_at", json_file.stem)
                snapshots.append((asset, ts, data))
            except Exception as e:
                print(f"  Warning: skipping {json_file}: {e}")

    return sorted(snapshots, key=lambda x: x[1])


def generate_fixtures(n_cycles=10, assets=None):
    """Generate synthetic multi-cycle fixtures for testing without Redis."""
    import random
    random.seed(42)
    assets = assets or ["GBPUSD", "USDJPY", "SPY"]
    snapshots = []

    for cycle in range(n_cycles):
        ts = f"2026-04-04T{11+cycle//6:02d}:{(cycle%6)*10:02d}:00+00:00"
        for asset in assets:
            is_fx = not asset.startswith("S")
            rsi = 50 + random.uniform(-20, 20) + cycle * random.uniform(-2, 2)
            rsi = max(20, min(80, rsi))
            macd = random.uniform(-0.005, 0.005)
            bb = random.uniform(0.1, 0.9)
            atr = random.uniform(0.005, 0.02) if is_fx else random.uniform(5, 15)
            price = random.uniform(1.2, 1.4) if is_fx else random.uniform(500, 700)

            technical = {
                "rsi": rsi, "macd_cross": macd,
                "bb_position_ratio": bb, "atr": atr, "price": price,
            }
            macro = {
                "carry_differential": random.uniform(-2, 4),
                "hours_to_next_event": random.uniform(-5, 48),
                "us10y_yield": 4.3 + random.uniform(-0.1, 0.1),
                "us2y_yield": 3.8 + random.uniform(-0.1, 0.1),
            }
            if is_fx:
                news = {
                    "base_bull_count": random.randint(3, 15),
                    "base_bear_count": random.randint(2, 12),
                    "base_article_count": random.randint(5, 20),
                    "quote_bull_count": random.randint(5, 20),
                    "quote_bear_count": random.randint(3, 18),
                    "quote_article_count": random.randint(8, 30),
                }
            else:
                news = {
                    "focus_bull_count": random.randint(3, 12),
                    "focus_bear_count": random.randint(2, 10),
                    "focus_article_count": random.randint(5, 15),
                    "market_bull_count": random.randint(5, 15),
                    "market_bear_count": random.randint(3, 12),
                    "market_article_count": random.randint(8, 20),
                }
            fundamental = {
                "base_cot_net_pct": random.uniform(-50, 50),
                "quote_cot_net_pct": random.uniform(-30, 30),
                "base_cot_net": random.randint(-50000, 50000),
            }
            risk = {
                "vix_level": 20 + random.uniform(-5, 10) + cycle * 0.3,
                "vix_change": random.uniform(-3, 3),
            }

            snapshot = {
                "captured_at": ts,
                "asset": asset,
                "capture_mode": "fixture",
                "agents_active": 6,
                "agents_total": 6,
                "agents": {
                    "technical": technical,
                    "macro": macro,
                    "news": news,
                    "fundamental": fundamental,
                    "risk_sentiment": risk,
                    "commodity": {"oil_price": 70 + random.uniform(-5, 5)},
                },
                "staleness": {k: random.uniform(5, 60) for k in
                              ["technical", "macro", "news", "fundamental",
                               "risk_sentiment", "commodity"]},
            }
            snapshots.append((asset, ts, snapshot))

    return snapshots


def process_multi_tick(snapshots, enrich=False, fred_key=None):
    """
    Process snapshots through stateful builders (one per asset).
    Returns per-tick feature evolution data.
    """
    validator = VectorValidator()
    builders = {}  # one per asset, maintains rolling state

    enricher = None
    if enrich:
        try:
            from prototypes.other.quantum_signal_gen1_infra.enrichment import MarketDataEnricher
            enricher = MarketDataEnricher(fred_api_key=fred_key)
            status = enricher.refresh()
            print(f"  Enrichment: yields={'OK' if status['yields_ok'] else 'FAIL'}, "
                  f"DXY={'OK' if status['dxy_ok'] else 'FAIL'}")
        except Exception as e:
            print(f"  Enrichment init failed: {e}")

    results = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "by_asset": defaultdict(lambda: {"vectors": [], "ticks": 0}),
        "feature_evolution": defaultdict(list),  # feature_name -> [(tick, value, asset)]
        "feature_first_active": {},  # feature_name -> tick number
        "enrichment_active": enricher is not None,
        "all_vectors": [],
    }

    tick_counter = defaultdict(int)  # per-asset tick number

    for asset, ts, snapshot in snapshots:
        if asset not in builders:
            builders[asset] = QuantumVectorBuilder(window_size=252)
        builder = builders[asset]

        agents = snapshot.get("agents", {})
        technical = agents.get("technical") or {}
        macro = agents.get("macro") or {}
        news = agents.get("news") or {}
        fundamental = agents.get("fundamental") or {}
        risk_sentiment = agents.get("risk_sentiment") or {}

        # Apply enrichment
        dxy_prices = None
        if enricher:
            macro = enricher.enrich_macro(macro)
            dxy_prices = enricher.get_dxy_prices_5d()

        # Skip snapshots with no real agent data
        has_data = any([technical, macro.get("carry_differential"),
                        news.get("base_bull_count") or news.get("focus_bull_count"),
                        fundamental.get("base_cot_net_pct")])
        if not has_data and not snapshot.get("capture_mode") == "fixture":
            continue

        try:
            vector = builder.build(
                technical=technical,
                macro=macro,
                news=news,
                fundamental=fundamental,
                risk_sentiment=risk_sentiment,
                dxy_prices_5d=dxy_prices,
                asset=asset,
                timestamp=snapshot.get("captured_at"),
            )

            report = validator.validate(vector)
            results["total"] += 1
            tick = tick_counter[asset]
            tick_counter[asset] += 1

            if report.is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1

            arr = vector.to_array()
            reduced = vector.to_reduced_8q()

            # Track feature evolution
            for i, (name, val) in enumerate(zip(FEATURE_NAMES, arr)):
                results["feature_evolution"][name].append({
                    "tick": tick, "value": val, "asset": asset, "ts": ts,
                })
                # Track first time feature becomes non-neutral
                is_neutral = (abs(val) < 1e-8 or
                              (name in ("vol_regime", "risk_sentiment") and abs(val - 0.5) < 1e-8))
                if not is_neutral and name not in results["feature_first_active"]:
                    results["feature_first_active"][name] = tick

            results["by_asset"][asset]["vectors"].append({
                "tick": tick,
                "ts": ts,
                "array_14q": arr,
                "array_8q": reduced,
                "valid": report.is_valid,
                "dict": vector.to_dict(),
            })
            results["by_asset"][asset]["ticks"] = tick + 1

            results["all_vectors"].append({
                "asset": asset, "tick": tick, "ts": ts,
                "array_14q": arr, "array_8q": reduced,
                "valid": report.is_valid,
            })

        except Exception as e:
            results["invalid"] += 1
            print(f"  Build error {asset} @ {ts}: {e}")

    return results


def print_evolution_report(results):
    """Print which features wake up and when."""
    print("\n" + "=" * 70)
    print("MULTI-TICK FEATURE EVOLUTION REPORT")
    print("=" * 70)
    print(f"\nTotal vectors built: {results['total']}")
    print(f"Valid: {results['valid']}  Invalid: {results['invalid']}")
    print(f"Assets: {list(results['by_asset'].keys())}")
    print(f"Enrichment: {'ACTIVE' if results['enrichment_active'] else 'OFF'}")

    max_ticks = max((d["ticks"] for d in results["by_asset"].values()), default=0)
    print(f"Max ticks per asset: {max_ticks}")

    print(f"\n{'─' * 70}")
    print("FEATURE ACTIVATION TIMELINE")
    print(f"{'─' * 70}")
    print(f"  {'Feature':<25s} {'First Active':>13s} {'Status':>10s}")
    print(f"  {'─'*25} {'─'*13} {'─'*10}")

    active_count = 0
    for name in FEATURE_NAMES:
        if name in results["feature_first_active"]:
            tick = results["feature_first_active"][name]
            status = "ACTIVE"
            active_count += 1
            print(f"  {name:<25s} {'tick ' + str(tick):>13s} {'ACTIVE':>10s}")
        else:
            entries = results["feature_evolution"].get(name, [])
            if not entries:
                reason = "no data"
            elif name in ("yield_curve_slope", "dxy_momentum"):
                reason = "needs enrichment"
            elif name in ("momentum_acceleration", "sentiment_momentum"):
                reason = f"needs ≥2 ticks"
            elif name in ("realized_vol_20d", "vol_regime", "carry_differential",
                          "risk_sentiment", "cot_extreme"):
                reason = f"needs history"
            else:
                reason = "neutral"
            print(f"  {name:<25s} {'—':>13s} {reason:>10s}")

    print(f"\n  Active: {active_count}/14 features")

    # 8-qubit reduced vector status
    print(f"\n{'─' * 70}")
    print("8-QUBIT REDUCED VECTOR STATUS")
    print(f"{'─' * 70}")
    active_8q = 0
    for name in REDUCED_8Q_NAMES:
        is_active = name in results["feature_first_active"]
        if is_active:
            active_8q += 1
        status = "ACTIVE" if is_active else "WARMING"
        print(f"  {name:<25s} {status:>10s}")
    print(f"\n  8Q active: {active_8q}/8 features")
    ready = active_8q >= 4
    print(f"  VQC ready: {'YES — enough signal diversity' if ready else 'NOT YET — need more ticks'}")

    return active_count, active_8q


def generate_evolution_html(results, output_path):
    """Generate an HTML report showing feature activation over ticks."""
    total = max(results["total"], 1)
    valid = results["valid"]
    active_count = len(results["feature_first_active"])

    # Build feature activation rows
    feature_rows = ""
    for name in FEATURE_NAMES:
        entries = results["feature_evolution"].get(name, [])
        if name in results["feature_first_active"]:
            tick = results["feature_first_active"][name]
            vals = [e["value"] for e in entries if abs(e["value"]) > 1e-8
                    and not (name in ("vol_regime", "risk_sentiment") and abs(e["value"] - 0.5) < 1e-8)]
            mn = min(vals) if vals else 0
            mx = max(vals) if vals else 0
            status = f'<span style="color:#4CAF50;font-weight:bold">ACTIVE (tick {tick})</span>'
            is_8q = "★" if name in REDUCED_8Q_NAMES else ""
        else:
            mn = mx = 0
            if name in ("yield_curve_slope", "dxy_momentum"):
                status = '<span style="color:#FF9800">needs enrichment</span>'
            elif name in ("momentum_acceleration", "sentiment_momentum"):
                status = '<span style="color:#2196F3">needs ≥2 ticks</span>'
            else:
                status = '<span style="color:#999">warming up</span>'
            is_8q = "★" if name in REDUCED_8Q_NAMES else ""

        feature_rows += f"""<tr>
            <td>{is_8q} {name}</td>
            <td style="font-family:monospace">{mn:.4f}</td>
            <td style="font-family:monospace">{mx:.4f}</td>
            <td>{status}</td>
        </tr>"""

    # Per-asset tick summary
    asset_rows = ""
    for asset, data in sorted(results["by_asset"].items()):
        n = data["ticks"]
        vectors = data["vectors"]
        valid_n = sum(1 for v in vectors if v["valid"])
        # Show last 8q vector
        if vectors:
            last_8q = vectors[-1]["array_8q"]
            nonzero = sum(1 for v in last_8q if abs(v) > 1e-8 and abs(v - 0.5) > 1e-8)
            last_8q_str = " ".join(f"{v:.3f}" for v in last_8q)
        else:
            nonzero = 0
            last_8q_str = "—"
        asset_rows += f"""<tr>
            <td>{asset}</td>
            <td>{n}</td>
            <td>{valid_n}/{n}</td>
            <td>{nonzero}/8 active</td>
            <td style="font-family:monospace;font-size:11px">{last_8q_str}</td>
        </tr>"""

    # Recent vector table
    vector_rows = ""
    for v in results["all_vectors"][-15:]:
        arr = v["array_8q"]
        cells = "".join(f'<td style="font-family:monospace;font-size:11px;'
                        f'background:{_heat(val)}">{val:.3f}</td>' for val in arr)
        vector_rows += f"""<tr>
            <td>{v['asset']}</td>
            <td>t{v['tick']}</td>
            {cells}
        </tr>"""

    active_8q = sum(1 for n in REDUCED_8Q_NAMES if n in results["feature_first_active"])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Phase 1A — Multi-Tick Validation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px;
         margin: 40px auto; padding: 0 20px; color: #333; background: #fafafa; }}
  h1 {{ border-bottom: 3px solid #1a1a2e; padding-bottom: 10px; color: #1a1a2e; }}
  h2 {{ color: #16213e; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; background: white; }}
  th, td {{ border: 1px solid #e0e0e0; padding: 7px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #1a1a2e; color: white; }}
  .stat {{ font-size: 32px; font-weight: bold; color: #1a1a2e; }}
  .stat-label {{ font-size: 12px; color: #777; text-transform: uppercase; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin: 20px 0; }}
  .stat-card {{ background: white; padding: 18px; border-radius: 8px; text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 10px;
            font-weight: bold; color: white; font-size: 12px; }}
  .badge-ok {{ background: #4CAF50; }}
  .badge-warn {{ background: #FF9800; }}
  .badge-info {{ background: #2196F3; }}
  .pipeline {{ display: flex; align-items: center; gap: 8px; margin: 20px 0;
               padding: 16px; background: white; border-radius: 8px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex-wrap: wrap; }}
  .pipeline-step {{ padding: 8px 16px; border-radius: 6px; font-size: 13px; font-weight: 500; }}
  .pipeline-arrow {{ color: #999; font-size: 18px; }}
  .step-done {{ background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }}
  .step-active {{ background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }}
  .step-next {{ background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }}
  footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd;
            color: #999; font-size: 12px; }}
</style></head>
<body>
<h1>Phase 1A — Quantum Signal Validation</h1>
<p style="color:#666">Multi-tick real-data validation &middot; Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<div class="pipeline">
  <div class="pipeline-step step-done">Docker Agents</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step step-done">Redis Pub/Sub</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step step-done">Capture</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step step-done">Builder (rolling)</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step step-done">14-Feature Vector</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step step-active">8Q Reduced</div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step step-next">VQC Smoke Test</div>
</div>

<div class="stat-grid">
  <div class="stat-card"><div class="stat">{results['total']}</div><div class="stat-label">Vectors Built</div></div>
  <div class="stat-card"><div class="stat" style="color:#4CAF50">{valid}</div><div class="stat-label">Valid ({100*valid/total:.0f}%)</div></div>
  <div class="stat-card"><div class="stat">{active_count}/14</div><div class="stat-label">Features Active</div></div>
  <div class="stat-card"><div class="stat">{active_8q}/8</div><div class="stat-label">8Q Features Active</div></div>
  <div class="stat-card"><div class="stat">{len(results['by_asset'])}</div><div class="stat-label">Assets</div></div>
</div>

<h2>Feature Activation Timeline</h2>
<p>★ = included in 8-qubit reduced vector</p>
<table>
<tr><th>Feature</th><th>Min</th><th>Max</th><th>Status</th></tr>
{feature_rows}
</table>

<h2>Per-Asset Summary</h2>
<table>
<tr><th>Asset</th><th>Ticks</th><th>Valid</th><th>8Q Signal</th><th>Last 8Q Vector</th></tr>
{asset_rows}
</table>

<h2>Recent 8-Qubit Vectors (heatmap)</h2>
<p style="font-size:12px;color:#666">Columns: momentum_composite, realized_vol_20d, vol_regime, carry_differential, event_risk_score, net_sentiment, cot_positioning, risk_sentiment</p>
<table>
<tr><th>Asset</th><th>Tick</th>
{"".join(f'<th style="font-size:11px">{n[:8]}</th>' for n in REDUCED_8Q_NAMES)}</tr>
{vector_rows}
</table>

<h2>VQC Readiness Assessment</h2>
<p>The 8-qubit reduced vector needs sufficient signal diversity for the VQC smoke test to be meaningful.
   With <strong>{active_8q}/8</strong> active features, the encoding will produce distinguishable quantum states
   across different market conditions.</p>
<p><strong>Assessment:</strong>
   {"<span class='badge badge-ok'>READY FOR VQC</span> — Enough signal diversity to produce meaningful angle encodings." if active_8q >= 3 else "<span class='badge badge-warn'>NEEDS MORE TICKS</span> — Continue capturing to build rolling history."}</p>

<footer>
  quantum_signal v1.0 &middot; QuantumInputVector contract FROZEN &middot; Phase 1A validation
</footer>
</body></html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nHTML report: {output_path}")


def _heat(val):
    """Simple heatmap color for a value in roughly [-1, 1]."""
    if abs(val) < 1e-8 or abs(val - 0.5) < 1e-8:
        return "#f5f5f5"  # neutral/zero
    if val > 0.3:
        return "#c8e6c9"
    if val > 0.1:
        return "#e8f5e9"
    if val < -0.3:
        return "#ffcdd2"
    if val < -0.1:
        return "#fce4ec"
    return "#fff9c4"


def main():
    parser = argparse.ArgumentParser(description="Multi-capture pipeline with rolling feature warm-up")
    parser.add_argument("--cycles", type=int, default=5, help="Number of capture cycles (default 5)")
    parser.add_argument("--interval", type=int, default=70, help="Seconds between captures (default 70)")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--skip-capture", action="store_true", help="Skip capture, process existing snapshots")
    parser.add_argument("--fixture-mode", action="store_true", help="Use synthetic fixtures (no Redis)")
    parser.add_argument("--enrich", action="store_true", help="Enable yield curve + DXY enrichment")
    parser.add_argument("--fred-key", help="FRED API key")
    parser.add_argument("--html", default=str(RESULTS_DIR / "phase1a_validation.html"),
                        help="HTML report output path")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 1A — MULTI-TICK VALIDATION PIPELINE")
    print("=" * 60)

    if args.fixture_mode:
        print(f"\nGenerating {args.cycles} synthetic fixture cycles...")
        snapshots = generate_fixtures(n_cycles=args.cycles)
        print(f"  Generated {len(snapshots)} fixture snapshots")
    elif not args.skip_capture:
        print(f"\nCapturing {args.cycles} cycles, {args.interval}s apart...")
        print(f"  Total time: ~{args.cycles * args.interval}s ({args.cycles * args.interval / 60:.1f} min)")
        print(f"  Redis: {args.redis_host}:{args.redis_port}")
        for i in range(args.cycles):
            print(f"\n  ── Cycle {i+1}/{args.cycles} ──")
            ok = run_capture(redis_port=args.redis_port, redis_host=args.redis_host)
            if not ok:
                print(f"  Capture {i+1} failed, continuing...")
            if i < args.cycles - 1:
                print(f"  Waiting {args.interval}s...")
                time.sleep(args.interval)

        snapshots = load_all_snapshots(SNAPSHOT_DIR)
        print(f"\nTotal snapshots on disk: {len(snapshots)}")
    else:
        snapshots = load_all_snapshots(SNAPSHOT_DIR)
        print(f"\nLoaded {len(snapshots)} existing snapshots")

    if not snapshots:
        print("ERROR: No snapshots available. Run with --fixture-mode or capture first.")
        sys.exit(1)

    # Process through stateful builder
    print("\nProcessing through canonical builder with rolling state...")
    results = process_multi_tick(
        snapshots, enrich=args.enrich, fred_key=args.fred_key
    )

    # Print report
    active_14, active_8q = print_evolution_report(results)

    # Generate HTML
    generate_evolution_html(results, args.html)

    # Save vector data for VQC smoke test
    vectors_path = RESULTS_DIR / "vectors_for_vqc.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    vqc_data = []
    for v in results["all_vectors"]:
        vqc_data.append({
            "asset": v["asset"],
            "tick": v["tick"],
            "features_8q": v["array_8q"],
            "features_14q": v["array_14q"],
            "valid": v["valid"],
        })
    with open(vectors_path, "w") as f:
        json.dump(vqc_data, f, indent=2)
    print(f"VQC input vectors saved: {vectors_path} ({len(vqc_data)} vectors)")

    print(f"\n{'=' * 60}")
    if active_8q >= 3:
        print("VERDICT: READY FOR VQC SMOKE TEST")
    else:
        print("VERDICT: NEED MORE CAPTURE TICKS")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
