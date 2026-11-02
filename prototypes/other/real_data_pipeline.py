#!/usr/bin/env python3
"""
Real-Data Pipeline
===================
End-to-end pipeline: captured snapshots → enrichment → vector build → validation.

This is the script that proves Phase 0 works on real captured data.

Usage:
  # Process all captured snapshots
  python scripts/real_data_pipeline.py --snapshot-dir snapshots/

  # Process a single asset
  python scripts/real_data_pipeline.py --snapshot-dir snapshots/ --asset EURUSD

  # With enrichment (fetch live yield curve + DXY)
  python scripts/real_data_pipeline.py --snapshot-dir snapshots/ --enrich

  # With enrichment using FRED API key
  FRED_API_KEY=your_key python scripts/real_data_pipeline.py --snapshot-dir snapshots/ --enrich

  # Generate HTML report
  python scripts/real_data_pipeline.py --snapshot-dir snapshots/ --enrich --html results/real_data_report.html

  # Process test fixtures as if they were snapshots (for dry-run testing)
  python scripts/real_data_pipeline.py --fixture-mode --enrich
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prototypes.other.quantum_signal_gen1_infra.builder import QuantumVectorBuilder
from prototypes.other.quantum_signal_gen1_infra.validator import VectorValidator
from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector

FEATURE_NAMES = [
    "momentum_composite", "momentum_acceleration", "realized_vol_20d",
    "vol_regime", "atr_normalized", "carry_differential",
    "yield_curve_slope", "event_risk_score", "net_sentiment",
    "sentiment_momentum", "cot_positioning", "cot_extreme",
    "dxy_momentum", "risk_sentiment",
]


def load_snapshots(snapshot_dir, asset_filter=None):
    """
    Load captured snapshots from disk.
    Returns list of (asset, timestamp, snapshot_dict) tuples, sorted by time.
    """
    snapshot_path = Path(snapshot_dir)
    if not snapshot_path.exists():
        print(f"  Snapshot directory not found: {snapshot_dir} (no captures yet)")
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
                print(f"  Warning: could not load {json_file}: {e}")

    return snapshots


def load_fixtures_as_snapshots():
    """
    Load test fixtures as if they were captured snapshots.
    Useful for dry-run testing the pipeline without Redis.
    """
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    if not fixtures_dir.exists():
        print(f"ERROR: Fixtures directory not found: {fixtures_dir}")
        return []

    # Load individual fixture files
    agents = {}
    file_map = {
        "technical": "sample_technical.json",
        "macro": "sample_macro.json",
        "news": "sample_news.json",
        "fundamental": "sample_fundamental.json",
        "risk_sentiment": "sample_risk_sentiment.json",
    }
    for key, fname in file_map.items():
        fpath = fixtures_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                agents[key] = json.load(f)

    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "asset": "EURUSD",
        "capture_mode": "fixture",
        "agents_active": len(agents),
        "agents_total": 5,
        "agents": agents,
        "staleness": {k: 0 for k in agents},
    }

    return [("EURUSD", snapshot["captured_at"], snapshot)]


def try_enrich(snapshot_data, enricher):
    """
    Attempt to enrich a snapshot with yield curve + DXY data.
    Returns the enriched snapshot dict and enrichment status.
    """
    from prototypes.other.quantum_signal_gen1_infra.enrichment import enrich_snapshot
    enriched = enrich_snapshot(snapshot_data.get("agents", {}), enricher)
    return enriched


def process_snapshots(snapshots, enrich=False, enricher=None):
    """
    Process all snapshots through the canonical builder + validator.
    Returns detailed results.
    """
    validator = VectorValidator()

    # One builder per asset to maintain proper rolling state
    builders = {}

    results = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "warnings_total": 0,
        "errors": [],
        "by_asset": defaultdict(lambda: {
            "total": 0, "valid": 0, "invalid": 0,
            "missing_agents": defaultdict(int),
            "stale_agents": defaultdict(int),
            "warnings": [],
        }),
        "feature_stats": defaultdict(lambda: {
            "min": float("inf"), "max": float("-inf"),
            "sum": 0, "count": 0, "zeros": 0,
        }),
        "enrichment_active": False,
        "yield_slope_alive": False,
        "dxy_alive": False,
        "vectors": [],  # store last N for viewer
    }

    for asset, ts, snapshot in snapshots:
        results["total"] += 1
        asset_r = results["by_asset"][asset]
        asset_r["total"] += 1

        if asset not in builders:
            builders[asset] = QuantumVectorBuilder(window_size=252)
        builder = builders[asset]

        agents = snapshot.get("agents", {})
        staleness = snapshot.get("staleness", {})

        # Track missing agents
        for agent_name in ["technical", "macro", "news", "fundamental"]:
            if not agents.get(agent_name):
                asset_r["missing_agents"][agent_name] += 1

        # Track stale agents (> 10 minutes old)
        for agent_name, age in staleness.items():
            if age is not None and age > 600:
                asset_r["stale_agents"][agent_name] += 1

        # Prepare builder inputs
        technical = agents.get("technical") or {}
        macro = agents.get("macro") or {}
        news = agents.get("news") or {}
        fundamental = agents.get("fundamental") or {}
        risk_sentiment = agents.get("risk_sentiment") or {}

        # Apply enrichment if available
        dxy_prices = None
        if enrich and enricher:
            from prototypes.other.quantum_signal_gen1_infra.enrichment import MarketDataEnricher
            macro = enricher.enrich_macro(macro)
            dxy_prices = enricher.get_dxy_prices_5d()
            results["enrichment_active"] = True

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

            if report.is_valid:
                results["valid"] += 1
                asset_r["valid"] += 1
            else:
                results["invalid"] += 1
                asset_r["invalid"] += 1
                results["errors"].append({
                    "asset": asset, "ts": ts, "errors": report.errors
                })

            results["warnings_total"] += len(report.warnings)
            asset_r["warnings"].extend(report.warnings)

            # Track feature distributions
            arr = vector.to_array()
            for i, (name, val) in enumerate(zip(FEATURE_NAMES, arr)):
                stats = results["feature_stats"][name]
                stats["min"] = min(stats["min"], val)
                stats["max"] = max(stats["max"], val)
                stats["sum"] += val
                stats["count"] += 1
                if abs(val) < 1e-8 or (name in ("vol_regime", "risk_sentiment") and abs(val - 0.5) < 1e-8):
                    stats["zeros"] += 1

            # Check if previously-dead features are now alive
            if abs(vector.yield_curve_slope) > 1e-8:
                results["yield_slope_alive"] = True
            if abs(vector.dxy_momentum) > 1e-8:
                results["dxy_alive"] = True

            # Store recent vectors for viewer
            if len(results["vectors"]) < 50:
                results["vectors"].append({
                    "asset": asset,
                    "timestamp": snapshot.get("captured_at"),
                    "vector_14q": vector.to_dict(),
                    "vector_8q": vector.to_reduced_8q(),
                    "agents_summary": {
                        k: bool(v) for k, v in agents.items()
                    },
                    "staleness": staleness,
                    "valid": report.is_valid,
                })

        except Exception as e:
            results["invalid"] += 1
            asset_r["invalid"] += 1
            results["errors"].append({
                "asset": asset, "ts": ts, "exception": str(e)
            })

    return results


def print_report(results):
    """Print human-readable report."""
    total = results["total"]
    if total == 0:
        print("No snapshots to process.")
        return "EMPTY"

    valid = results["valid"]
    invalid = results["invalid"]

    print("=" * 70)
    print("REAL-DATA PIPELINE REPORT")
    print("=" * 70)
    print(f"\nTotal snapshots processed: {total}")
    print(f"Valid vectors:            {valid} ({100*valid/total:.1f}%)")
    print(f"Invalid vectors:          {invalid}")
    print(f"Total warnings:           {results['warnings_total']}")
    print(f"Enrichment active:        {'YES' if results['enrichment_active'] else 'NO'}")

    print(f"\n{'─' * 70}")
    print("PREVIOUSLY-DEAD FEATURES")
    print(f"{'─' * 70}")
    ys = "ALIVE" if results["yield_slope_alive"] else "STILL ZERO (no data source)"
    dx = "ALIVE" if results["dxy_alive"] else "STILL ZERO (no data source)"
    print(f"  yield_curve_slope: {ys}")
    print(f"  dxy_momentum:      {dx}")

    print(f"\n{'─' * 70}")
    print("PER-ASSET BREAKDOWN")
    print(f"{'─' * 70}")
    for asset, data in sorted(results["by_asset"].items()):
        status = "PASS" if data["invalid"] == 0 else "FAIL"
        missing = dict(data["missing_agents"])
        stale = dict(data["stale_agents"])
        print(f"  {asset:8s}  {data['valid']}/{data['total']} valid [{status}]"
              f"  missing={missing or 'none'}"
              f"  stale={stale or 'none'}")

    print(f"\n{'─' * 70}")
    print("FEATURE DISTRIBUTION")
    print(f"{'─' * 70}")
    print(f"  {'Feature':<25s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Zero%':>8s} {'Active':>8s}")
    for name in FEATURE_NAMES:
        stats = results["feature_stats"].get(name, {})
        n = stats.get("count", 0)
        if n == 0:
            continue
        mn = stats["min"]
        mx = stats["max"]
        mean = stats["sum"] / n
        zero_rate = stats["zeros"] / n
        active = "YES" if mn != mx else "NO"
        print(f"  {name:<25s} {mn:>8.4f} {mx:>8.4f} {mean:>8.4f} {100*zero_rate:>7.1f}% {active:>8s}")

    if results["errors"]:
        print(f"\n{'─' * 70}")
        print(f"ERRORS ({len(results['errors'])})")
        for err in results["errors"][:5]:
            print(f"  {err}")

    print(f"\n{'=' * 70}")
    if invalid == 0 and total > 0:
        verdict = "PASS"
        if not results["yield_slope_alive"] or not results["dxy_alive"]:
            verdict = "PASS WITH KNOWN GAPS"
    elif invalid < total * 0.02:
        verdict = "PASS WITH WARNINGS"
    else:
        verdict = "FAIL"
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 70}")
    return verdict


def generate_html_report(results, output_path):
    """Generate a static HTML report from pipeline results."""
    total = max(results["total"], 1)  # guard against division by zero
    valid = results["valid"]

    # Build feature table rows
    feature_rows = ""
    for name in FEATURE_NAMES:
        stats = results["feature_stats"].get(name, {})
        n = stats.get("count", 0)
        if n == 0:
            continue
        mn = stats["min"]
        mx = stats["max"]
        mean = stats["sum"] / n
        zero_rate = stats["zeros"] / n
        pct = ((mean - mn) / (mx - mn)) * 100 if mx != mn else 50
        alive = mn != mx
        color = "#4CAF50" if alive else "#F44336"
        feature_rows += f"""<tr>
            <td>{name}</td>
            <td style="font-family:monospace">{mn:.4f}</td>
            <td style="font-family:monospace">{mx:.4f}</td>
            <td style="font-family:monospace">{mean:.4f}</td>
            <td>{100*zero_rate:.1f}%</td>
            <td style="color:{color};font-weight:bold">{'YES' if alive else 'NO'}</td>
        </tr>"""

    # Asset rows
    asset_rows = ""
    for asset, data in sorted(results["by_asset"].items()):
        status_color = "#4CAF50" if data["invalid"] == 0 else "#F44336"
        asset_rows += f"""<tr>
            <td>{asset}</td>
            <td>{data['valid']}/{data['total']}</td>
            <td style="color:{status_color};font-weight:bold">{'PASS' if data['invalid']==0 else 'FAIL'}</td>
            <td>{dict(data['missing_agents']) or '—'}</td>
            <td>{dict(data['stale_agents']) or '—'}</td>
        </tr>"""

    # Vector detail rows (last 10)
    vector_rows = ""
    for v in results.get("vectors", [])[-10:]:
        d = v["vector_14q"]
        vector_rows += f"""<tr>
            <td>{v['asset']}</td>
            <td style="font-size:11px">{v['timestamp'][:19]}</td>
            <td style="font-family:monospace;font-size:11px">{d.get('momentum_composite',0):.3f}</td>
            <td style="font-family:monospace;font-size:11px">{d.get('carry_differential',0):.3f}</td>
            <td style="font-family:monospace;font-size:11px">{d.get('yield_curve_slope',0):.3f}</td>
            <td style="font-family:monospace;font-size:11px">{d.get('net_sentiment',0):.3f}</td>
            <td style="font-family:monospace;font-size:11px">{d.get('dxy_momentum',0):.3f}</td>
            <td>{'Valid' if v['valid'] else 'INVALID'}</td>
        </tr>"""

    ys_badge = ('<span class="badge ok">ALIVE</span>' if results["yield_slope_alive"]
                else '<span class="badge warn">ZERO</span>')
    dx_badge = ('<span class="badge ok">ALIVE</span>' if results["dxy_alive"]
                else '<span class="badge warn">ZERO</span>')
    enrich_badge = ('<span class="badge ok">ON</span>' if results["enrichment_active"]
                    else '<span class="badge off">OFF</span>')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Phase 0 Real-Data Pipeline Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ color: #555; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #f5f5f5; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 10px; font-weight: bold; color: white; font-size: 12px; margin: 0 4px; }}
  .badge.ok {{ background: #4CAF50; }}
  .badge.warn {{ background: #FF9800; }}
  .badge.off {{ background: #999; }}
  .stat {{ font-size: 28px; font-weight: bold; }}
  .stat-label {{ font-size: 12px; color: #777; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .stat-card {{ background: #f9f9f9; padding: 16px; border-radius: 8px; text-align: center; }}
</style></head>
<body>
<h1>Phase 0 — Real-Data Pipeline Report</h1>
<p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<div class="stat-grid">
  <div class="stat-card"><div class="stat">{total}</div><div class="stat-label">Snapshots</div></div>
  <div class="stat-card"><div class="stat" style="color:#4CAF50">{valid}</div><div class="stat-label">Valid ({100*valid/total:.0f}%)</div></div>
  <div class="stat-card"><div class="stat">{results['warnings_total']}</div><div class="stat-label">Warnings</div></div>
  <div class="stat-card"><div class="stat">{len(results['by_asset'])}</div><div class="stat-label">Assets</div></div>
</div>

<p>Enrichment: {enrich_badge} &nbsp; yield_curve_slope: {ys_badge} &nbsp; dxy_momentum: {dx_badge}</p>

<h2>Per-Asset Results</h2>
<table><tr><th>Asset</th><th>Valid</th><th>Status</th><th>Missing Agents</th><th>Stale Agents</th></tr>
{asset_rows}</table>

<h2>Feature Distribution</h2>
<table><tr><th>Feature</th><th>Min</th><th>Max</th><th>Mean</th><th>Zero%</th><th>Active</th></tr>
{feature_rows}</table>

<h2>Recent Vectors (last 10)</h2>
<table><tr><th>Asset</th><th>Timestamp</th><th>Momentum</th><th>Carry</th><th>Yield</th><th>Sentiment</th><th>DXY</th><th>Valid</th></tr>
{vector_rows}</table>

<p style="color:#999;font-size:12px;margin-top:40px">Phase 0 Real-Data Pipeline — quantum_signal v1.0</p>
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-Data Pipeline for Phase 0 validation")
    parser.add_argument("--snapshot-dir", help="Directory with captured snapshots")
    parser.add_argument("--asset", help="Process single asset only")
    parser.add_argument("--enrich", action="store_true", help="Enable yield curve + DXY enrichment")
    parser.add_argument("--fred-key", help="FRED API key (or set FRED_API_KEY env var)")
    parser.add_argument("--fixture-mode", action="store_true", help="Use test fixtures instead of real snapshots")
    parser.add_argument("--html", help="Output HTML report path")
    parser.add_argument("--json-out", help="Output JSON results path")
    args = parser.parse_args()

    # Load snapshots
    if args.snapshot_dir:
        print(f"Loading snapshots from: {args.snapshot_dir}")
        snapshots = load_snapshots(args.snapshot_dir, asset_filter=args.asset)
        if not snapshots and not args.fixture_mode:
            print("  No snapshots found. Falling back to fixture mode.")
            print("  (Use --fixture-mode explicitly to skip this message.)")
            args.fixture_mode = True
    else:
        snapshots = []

    if args.fixture_mode:
        print("Loading test fixtures as snapshots...")
        snapshots = load_fixtures_as_snapshots()

    if not snapshots and not args.fixture_mode:
        print("ERROR: provide --snapshot-dir or --fixture-mode")
        sys.exit(1)

    print(f"Loaded {len(snapshots)} snapshots")

    # Setup enrichment if requested
    enricher = None
    if args.enrich:
        from prototypes.other.quantum_signal_gen1_infra.enrichment import MarketDataEnricher
        enricher = MarketDataEnricher(
            fred_api_key=args.fred_key,
        )
        print("Refreshing enrichment data (yield curve + DXY)...")
        status = enricher.refresh()
        print(f"  Yields: {'OK' if status['yields_ok'] else 'FAILED'}"
              f" (10Y={status.get('us10y')}, 2Y={status.get('us2y')}, spread={status.get('spread')})")
        print(f"  DXY: {'OK' if status['dxy_ok'] else 'FAILED'}"
              f" ({status.get('dxy_prices_count', 0)} prices)")
        if status["errors"]:
            for e in status["errors"]:
                print(f"  Warning: {e}")

    # Process
    results = process_snapshots(snapshots, enrich=args.enrich, enricher=enricher)
    verdict = print_report(results)

    # Save outputs
    if args.html and results["total"] > 0:
        os.makedirs(os.path.dirname(args.html) or ".", exist_ok=True)
        generate_html_report(results, args.html)
    elif args.html and results["total"] == 0:
        print(f"Skipping HTML report (no snapshots to report on).")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        # Clean up non-serializable parts
        clean_results = dict(results)
        clean_results["feature_stats"] = {
            k: dict(v) for k, v in results["feature_stats"].items()
        }
        clean_results["by_asset"] = {
            k: {
                "total": v["total"], "valid": v["valid"], "invalid": v["invalid"],
                "missing_agents": dict(v["missing_agents"]),
                "stale_agents": dict(v["stale_agents"]),
            }
            for k, v in results["by_asset"].items()
        }
        with open(args.json_out, "w") as f:
            json.dump(clean_results, f, indent=2, default=str)
        print(f"JSON results saved to: {args.json_out}")


if __name__ == "__main__":
    main()
