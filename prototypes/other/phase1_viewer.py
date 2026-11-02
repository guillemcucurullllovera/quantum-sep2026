#!/usr/bin/env python3
"""
Phase 1 Readable Layer — Minimal Terminal Dashboard
=====================================================
The thinnest possible interface to verify the quantum_signal pipeline is alive.

Shows:
  - Raw agent payload summary (what came in)
  - Transformed 14-feature vector (what the builder produced)
  - Reduced 8-feature vector (what goes to the simulator)
  - Validation / health / staleness status
  - VQC readiness indicator

Usage:
  # From fixture files:
  python scripts/phase1_viewer.py --input-dir tests/fixtures/ --asset EURUSD

  # From a single JSON snapshot (all agents combined):
  python scripts/phase1_viewer.py --snapshot path/to/snapshot.json

  # Generate N sequential ticks and show dashboard for each:
  python scripts/phase1_viewer.py --simulate --asset EURUSD --ticks 5

  # Static HTML output:
  python scripts/phase1_viewer.py --input-dir tests/fixtures/ --asset EURUSD --html report.html
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prototypes.other.quantum_signal_gen1_infra.builder import QuantumVectorBuilder
from prototypes.other.quantum_signal_gen1_infra.reader import SignalReader
from prototypes.other.quantum_signal_gen1_infra.validator import VectorValidator
from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector

FEATURE_NAMES = [
    "momentum_composite", "momentum_acceleration", "realized_vol_20d",
    "vol_regime", "atr_normalized", "carry_differential",
    "yield_curve_slope", "event_risk_score", "net_sentiment",
    "sentiment_momentum", "cot_positioning", "cot_extreme",
    "dxy_momentum", "risk_sentiment",
]

FEATURE_RANGES = {
    "momentum_composite": (-1, 1), "momentum_acceleration": (-1, 1),
    "realized_vol_20d": (0, 1), "vol_regime": (0, 1),
    "atr_normalized": (0, 1), "carry_differential": (-1, 1),
    "yield_curve_slope": (-1, 1), "event_risk_score": (0, 1),
    "net_sentiment": (-1, 1), "sentiment_momentum": (-1, 1),
    "cot_positioning": (-1, 1), "cot_extreme": (-1, 1),
    "dxy_momentum": (-1, 1), "risk_sentiment": (0, 1),
}

QUBIT_8Q_INDICES = [0, 2, 3, 5, 7, 8, 10, 13]
QUBIT_8Q_NAMES = [FEATURE_NAMES[i] for i in QUBIT_8Q_INDICES]


def bar_chart(value, lo, hi, width=20):
    """Render a small inline bar for a feature value."""
    if hi == lo:
        pos = width // 2
    else:
        normalized = (value - lo) / (hi - lo)
        pos = int(normalized * width)
    pos = max(0, min(width, pos))
    if lo < 0:  # bipolar: center is neutral
        mid = width // 2
        bar = list("·" * width)
        if pos < mid:
            for i in range(pos, mid):
                bar[i] = "▓"
        else:
            for i in range(mid, pos):
                bar[i] = "▓"
        return "".join(bar)
    else:  # unipolar: left is 0
        return "▓" * pos + "·" * (width - pos)


def summarize_payload(name, payload):
    """Compact summary of a raw agent payload."""
    if not payload:
        return f"  {name:14s}: [MISSING]"
    keys = [k for k in payload.keys() if k not in ("agent_id", "asset", "asset_type", "signal_type", "source", "timestamp", "weights_used", "top_headlines")]
    parts = []
    for k in keys[:6]:
        v = payload[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, int):
            parts.append(f"{k}={v}")
        elif isinstance(v, bool):
            parts.append(f"{k}={'T' if v else 'F'}")
        elif isinstance(v, str):
            parts.append(f"{k}={v[:15]}")
        elif v is None:
            parts.append(f"{k}=null")
    overflow = f" +{len(keys)-6} more" if len(keys) > 6 else ""
    return f"  {name:14s}: {', '.join(parts)}{overflow}"


def health_check(vector, payloads, timestamp_str):
    """Assess vector health: staleness, completeness, readiness."""
    issues = []
    info = []

    # Check timestamp freshness
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if age_min > 60:
            issues.append(f"STALE: vector is {age_min:.0f}min old")
        elif age_min > 15:
            info.append(f"age: {age_min:.0f}min")
    except Exception:
        info.append("timestamp: unparseable")

    # Check for missing agent data
    for name in ["technical", "macro", "news", "fundamental"]:
        p = payloads.get(name, {})
        if not p or len(p) <= 1:  # empty or just timestamp
            issues.append(f"MISSING: {name} agent data")

    # Check features at neutral/default
    arr = vector.to_array()
    neutral_count = 0
    always_neutral = []
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, arr)):
        lo, hi = FEATURE_RANGES[name]
        mid = (lo + hi) / 2
        if abs(val - mid) < 1e-6 or abs(val) < 1e-8:
            neutral_count += 1
            always_neutral.append(name)
    if neutral_count > 4:
        info.append(f"{neutral_count}/14 features at neutral (cold-start likely)")

    # Known gaps
    if abs(vector.yield_curve_slope) < 1e-8:
        info.append("yield_curve_slope: no data source (known gap)")
    if abs(vector.dxy_momentum) < 1e-8:
        info.append("dxy_momentum: no data source (known gap)")

    # VQC readiness
    active_features_14q = sum(1 for v in arr if abs(v) > 1e-8 and abs(v - 0.5) > 1e-8)
    reduced = vector.to_reduced_8q()
    active_features_8q = sum(1 for v in reduced if abs(v) > 1e-8 and abs(v - 0.5) > 1e-8)

    return {
        "issues": issues,
        "info": info,
        "neutral_features": always_neutral,
        "active_14q": active_features_14q,
        "active_8q": active_features_8q,
        "vqc_ready": len(issues) == 0 and active_features_8q >= 4,
        "qaoa_ready": True,  # QAOA is asset-pair based, doesn't need features
    }


def print_dashboard(vector, payloads, health, tick_num=None):
    """Print the terminal dashboard."""
    w = 70
    tick_label = f" (tick #{tick_num})" if tick_num is not None else ""
    print(f"\n{'═' * w}")
    print(f"  QUANTUM SIGNAL VIEWER — {vector.asset}{tick_label}")
    print(f"  {vector.timestamp}")
    print(f"{'═' * w}")

    # Raw payloads
    print(f"\n  ┌{'─' * (w-4)}┐")
    print(f"  │ {'RAW AGENT PAYLOADS':<{w-5}}│")
    print(f"  ├{'─' * (w-4)}┤")
    for name in ["technical", "macro", "news", "fundamental", "risk_sentiment"]:
        line = summarize_payload(name, payloads.get(name, {}))
        print(f"  │{line:<{w-4}}│")
    print(f"  └{'─' * (w-4)}┘")

    # 14-feature vector
    arr = vector.to_array()
    print(f"\n  ┌{'─' * (w-4)}┐")
    print(f"  │ {'14-QUBIT VECTOR':<{w-5}}│")
    print(f"  ├{'─' * (w-4)}┤")
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, arr)):
        lo, hi = FEATURE_RANGES[name]
        bar = bar_chart(val, lo, hi, width=16)
        marker = " ◄8q" if i in QUBIT_8Q_INDICES else ""
        line = f"  q{i:02d} {name:<25s} {val:>8.4f}  [{bar}]{marker}"
        print(f"  │{line[2:]:<{w-4}}│")
    print(f"  └{'─' * (w-4)}┘")

    # 8-feature reduced vector
    reduced = vector.to_reduced_8q()
    print(f"\n  ┌{'─' * (w-4)}┐")
    print(f"  │ {'8-QUBIT REDUCED VECTOR (simulator-friendly)':<{w-5}}│")
    print(f"  ├{'─' * (w-4)}┤")
    for j, (idx, name, val) in enumerate(zip(QUBIT_8Q_INDICES, QUBIT_8Q_NAMES, reduced)):
        lo, hi = FEATURE_RANGES[name]
        bar = bar_chart(val, lo, hi, width=16)
        ry_angle = val * math.pi
        line = f"  q{j} (←q{idx:02d}) {name:<22s} {val:>7.4f}  RY={ry_angle:>6.3f}  [{bar}]"
        print(f"  │{line[2:]:<{w-4}}│")
    print(f"  └{'─' * (w-4)}┘")

    # Health / status
    print(f"\n  ┌{'─' * (w-4)}┐")
    print(f"  │ {'HEALTH & READINESS':<{w-5}}│")
    print(f"  ├{'─' * (w-4)}┤")
    status_14 = f"14q active features: {health['active_14q']}/14"
    status_8 = f" 8q active features: {health['active_8q']}/8"
    vqc_icon = "● READY" if health["vqc_ready"] else "○ NOT READY"
    qaoa_icon = "● READY" if health["qaoa_ready"] else "○ NOT READY"
    print(f"  │  {status_14:<{w-6}}│")
    print(f"  │  {status_8:<{w-6}}│")
    print(f"  │  VQC input:  {vqc_icon:<{w-17}}│")
    print(f"  │  QAOA input: {qaoa_icon:<{w-17}}│")

    if health["issues"]:
        print(f"  │{'':>{w-4}}│")
        for issue in health["issues"]:
            line = f"  ⚠ {issue}"
            print(f"  │{line:<{w-4}}│")
    if health["info"]:
        for info in health["info"]:
            line = f"  ℹ {info}"
            print(f"  │{line:<{w-4}}│")

    print(f"  └{'─' * (w-4)}┘")


def generate_html(vector, payloads, health):
    """Generate a static HTML report."""
    arr = vector.to_array()
    reduced = vector.to_reduced_8q()

    features_rows = ""
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, arr)):
        lo, hi = FEATURE_RANGES[name]
        pct = ((val - lo) / (hi - lo)) * 100 if hi != lo else 50
        in_8q = "★" if i in QUBIT_8Q_INDICES else ""
        color = "#4CAF50" if abs(val) > 0.1 else ("#FFC107" if abs(val) > 0.01 else "#999")
        features_rows += f"""
        <tr>
          <td>q{i:02d}</td>
          <td>{name}</td>
          <td style="text-align:right;font-family:monospace">{val:.4f}</td>
          <td><div style="background:#eee;width:200px;height:18px;position:relative;border-radius:3px">
            <div style="background:{color};width:{pct:.0f}%;height:100%;border-radius:3px"></div>
          </div></td>
          <td style="text-align:center">{in_8q}</td>
        </tr>"""

    reduced_rows = ""
    for j, (idx, name, val) in enumerate(zip(QUBIT_8Q_INDICES, QUBIT_8Q_NAMES, reduced)):
        ry = val * math.pi
        reduced_rows += f"""
        <tr>
          <td>q{j} (←q{idx:02d})</td>
          <td>{name}</td>
          <td style="font-family:monospace">{val:.4f}</td>
          <td style="font-family:monospace">RY={ry:.3f}</td>
        </tr>"""

    vqc_status = "READY" if health["vqc_ready"] else "NOT READY"
    vqc_color = "#4CAF50" if health["vqc_ready"] else "#F44336"
    issues_html = "".join(f"<li style='color:#F44336'>{i}</li>" for i in health["issues"])
    info_html = "".join(f"<li style='color:#666'>{i}</li>" for i in health["info"])

    raw_payloads_html = ""
    for name in ["technical", "macro", "news", "fundamental", "risk_sentiment"]:
        p = payloads.get(name, {})
        display = {k: v for k, v in p.items() if k not in ("agent_id", "asset", "asset_type", "signal_type", "source", "weights_used", "top_headlines")}
        raw_payloads_html += f"""
        <details><summary><strong>{name}</strong> ({len(display)} fields)</summary>
        <pre style="background:#f5f5f5;padding:8px;border-radius:4px;font-size:12px">{json.dumps(display, indent=2)}</pre>
        </details>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Quantum Signal Viewer — {vector.asset}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ color: #555; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #f5f5f5; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-weight: bold; color: white; font-size: 14px; }}
  details {{ margin: 4px 0; }}
  summary {{ cursor: pointer; padding: 4px 0; }}
</style></head>
<body>
<h1>Quantum Signal Viewer</h1>
<p><strong>Asset:</strong> {vector.asset} &nbsp; <strong>Timestamp:</strong> {vector.timestamp} &nbsp; <strong>Version:</strong> {vector.version}</p>
<p><span class="badge" style="background:{vqc_color}">VQC: {vqc_status}</span>
   <span class="badge" style="background:#4CAF50">QAOA: READY</span>
   <span class="badge" style="background:#2196F3">{health['active_14q']}/14 active</span>
   <span class="badge" style="background:#2196F3">{health['active_8q']}/8 reduced active</span></p>

<h2>Raw Agent Payloads</h2>
{raw_payloads_html}

<h2>14-Qubit Vector</h2>
<table>
<tr><th>Qubit</th><th>Feature</th><th>Value</th><th>Distribution</th><th>8q</th></tr>
{features_rows}
</table>

<h2>8-Qubit Reduced Vector (Simulator Input)</h2>
<table>
<tr><th>Qubit</th><th>Feature</th><th>Value</th><th>RY Angle</th></tr>
{reduced_rows}
</table>

<h2>Health & Status</h2>
<ul>
{issues_html}
{info_html}
</ul>

<p style="color:#999;font-size:12px;margin-top:40px">Generated by Phase 1 Readable Layer — quantum_signal v{vector.version}</p>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Quantum Signal Viewer")
    parser.add_argument("--input-dir", help="Directory with agent fixture JSON files")
    parser.add_argument("--snapshot", help="Single JSON file with all agent payloads")
    parser.add_argument("--asset", default="EURUSD", help="Asset identifier")
    parser.add_argument("--simulate", action="store_true", help="Generate simulated ticks")
    parser.add_argument("--ticks", type=int, default=1, help="Number of ticks for --simulate")
    parser.add_argument("--html", help="Output HTML file path")
    parser.add_argument("--json-out", help="Output JSON file path")
    parser.add_argument("--enrich", action="store_true", help="Fetch live yield curve + DXY data")
    parser.add_argument("--fred-key", help="FRED API key (or set FRED_API_KEY env var)")
    args = parser.parse_args()

    builder = QuantumVectorBuilder(window_size=252)
    validator = VectorValidator()

    # Setup enrichment if requested
    enricher = None
    dxy_prices = None
    if args.enrich:
        try:
            from prototypes.other.quantum_signal_gen1_infra.enrichment import MarketDataEnricher
            enricher = MarketDataEnricher(fred_api_key=args.fred_key)
            status = enricher.refresh()
            print(f"Enrichment: Yields={'OK' if status['yields_ok'] else 'FAIL'}"
                  f" (10Y={status.get('us10y')}, 2Y={status.get('us2y')})"
                  f"  DXY={'OK' if status['dxy_ok'] else 'FAIL'}"
                  f" ({status.get('dxy_prices_count',0)} prices)")
            dxy_prices = enricher.get_dxy_prices_5d()
        except Exception as e:
            print(f"Enrichment failed: {e}")

    if args.simulate:
        # Import the simulation generators
        from real_system_validation import (
            generate_technical, generate_macro, generate_news,
            generate_fundamental, generate_risk_sentiment,
        )
        import random
        from datetime import timedelta
        random.seed(99)
        base_t = datetime(2026, 4, 4, 9, 0, 0, tzinfo=timezone.utc)

        for tick in range(args.ticks):
            t = base_t + timedelta(minutes=tick * 15)
            payloads = {
                "technical": generate_technical(args.asset, t, tick),
                "macro": generate_macro(args.asset, t, tick),
                "news": generate_news(args.asset, t, tick),
                "fundamental": generate_fundamental(args.asset, t, tick),
                "risk_sentiment": generate_risk_sentiment(t),
            }
            vector = builder.build(
                technical=payloads["technical"],
                macro=payloads["macro"],
                news=payloads["news"],
                fundamental=payloads["fundamental"],
                risk_sentiment=payloads["risk_sentiment"],
                asset=args.asset,
                timestamp=t.isoformat() + "Z",
            )
            health = health_check(vector, payloads, t.isoformat() + "Z")
            print_dashboard(vector, payloads, health, tick_num=tick)

            if args.html and tick == args.ticks - 1:
                html = generate_html(vector, payloads, health)
                with open(args.html, "w") as f:
                    f.write(html)
                print(f"\nHTML saved to: {args.html}")

    elif args.input_dir:
        # Load individual fixture files directly
        import os as _os
        payloads = {}
        file_map = {
            "technical": "sample_technical.json",
            "macro": "sample_macro.json",
            "news": "sample_news.json",
            "fundamental": "sample_fundamental.json",
            "risk_sentiment": "sample_risk_sentiment.json",
        }
        for key, fname in file_map.items():
            fpath = _os.path.join(args.input_dir, fname)
            if _os.path.exists(fpath):
                with open(fpath) as f:
                    payloads[key] = json.load(f)
            else:
                payloads[key] = {}

        timestamp = datetime.now(timezone.utc).isoformat()

        # Apply enrichment if available
        macro = payloads.get("macro", {})
        if enricher:
            macro = enricher.enrich_macro(macro)

        vector = builder.build(
            technical=payloads.get("technical", {}),
            macro=macro,
            news=payloads.get("news", {}),
            fundamental=payloads.get("fundamental", {}),
            risk_sentiment=payloads.get("risk_sentiment", {}),
            dxy_prices_5d=dxy_prices,
            asset=args.asset,
            timestamp=timestamp,
        )
        health = health_check(vector, payloads, timestamp)
        print_dashboard(vector, payloads, health)

        if args.html:
            html = generate_html(vector, payloads, health)
            with open(args.html, "w") as f:
                f.write(html)
            print(f"\nHTML saved to: {args.html}")

        if args.json_out:
            output = {
                "vector_14q": vector.to_dict(),
                "vector_8q": vector.to_reduced_8q(),
                "health": health,
                "raw_payloads": {k: v for k, v in payloads.items()},
            }
            with open(args.json_out, "w") as f:
                json.dump(output, f, indent=2, default=str)
            print(f"JSON saved to: {args.json_out}")

    elif args.snapshot:
        with open(args.snapshot) as f:
            raw = json.load(f)
        # Handle captured snapshot format (has "agents" key)
        # vs direct payload format
        if "agents" in raw:
            payloads = raw["agents"]
            timestamp = raw.get("captured_at", datetime.now(timezone.utc).isoformat())
            asset = raw.get("asset", args.asset)
        else:
            payloads = raw
            timestamp = datetime.now(timezone.utc).isoformat()
            asset = args.asset

        macro = payloads.get("macro") or {}
        if enricher:
            macro = enricher.enrich_macro(macro)

        vector = builder.build(
            technical=payloads.get("technical") or {},
            macro=macro,
            news=payloads.get("news") or {},
            fundamental=payloads.get("fundamental") or {},
            risk_sentiment=payloads.get("risk_sentiment") or {},
            dxy_prices_5d=dxy_prices,
            asset=asset,
            timestamp=timestamp,
        )
        health = health_check(vector, payloads, timestamp)
        print_dashboard(vector, payloads, health)

        if args.html:
            html = generate_html(vector, payloads, health)
            with open(args.html, "w") as f:
                f.write(html)
            print(f"\nHTML saved to: {args.html}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
