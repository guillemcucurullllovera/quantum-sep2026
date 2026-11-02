#!/usr/bin/env python3
"""
Reprocess features using fixed parsers and wider grouping window.

This script works offline (no Yahoo Finance needed) by:
1. Re-grouping signals from JSONL with 600s window (captures all 4 agent types)
2. Re-parsing with fixed technical parser (handles lowercase rsi/macd/bb/atr)
3. Running through QuantumVectorBuilder for proper feature computation
4. Matching to existing price/label data by (asset, closest timestamp)

Usage:
    python scripts/reprocess_features.py
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from dateutil.parser import isoparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from prototypes.other.quantum_signal_gen1_infra.builder import QuantumVectorBuilder
from prototypes.other.signals_to_dataset import (
    group_into_snapshots, load_signals, parse_timestamp,
    compute_price_momentum_4h, compute_intraday_vol,
    FAST_FEATURE_SLOT_PRICE_MOM, FAST_FEATURE_SLOT_INTRADAY_VOL,
)

PHASE0_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PHASE0_DIR, "results")
SIGNALS_FILE = os.path.join(PHASE0_DIR, "..", "signals_export.jsonl")


def build_price_lookup(existing_dataset_path):
    """Build a lookup table from existing dataset: (asset, timestamp) -> {price, fwd_return, label}."""
    with open(existing_dataset_path) as f:
        data = json.load(f)

    samples = data.get("train", []) + data.get("val", [])
    print(f"  Loaded {len(samples)} existing samples for price lookup")

    # Index by (asset, timestamp_epoch)
    lookup = defaultdict(list)
    for s in samples:
        asset = s.get("asset", "?")
        ts = isoparse(s["timestamp"])
        epoch = ts.timestamp()
        lookup[asset].append({
            "epoch": epoch,
            "price": s["price"],
            "fwd_return": s["fwd_return"],
            "label": s["label"],
            "timestamp": s["timestamp"],
        })

    # Sort by epoch for binary search
    for asset in lookup:
        lookup[asset].sort(key=lambda x: x["epoch"])

    return dict(lookup)


def find_closest_price(lookup, asset, target_ts, max_gap_seconds=600):
    """Find the closest price entry within max_gap_seconds."""
    entries = lookup.get(asset, [])
    if not entries:
        return None

    target_epoch = target_ts.timestamp()

    # Binary search
    lo, hi = 0, len(entries) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if entries[mid]["epoch"] < target_epoch:
            lo = mid + 1
        else:
            hi = mid

    # Check lo and lo-1
    best = None
    best_gap = float("inf")
    for idx in [lo - 1, lo, lo + 1]:
        if 0 <= idx < len(entries):
            gap = abs(entries[idx]["epoch"] - target_epoch)
            if gap < best_gap:
                best_gap = gap
                best = entries[idx]

    if best and best_gap <= max_gap_seconds:
        return best
    return None


def count_alive(features):
    return sum(1 for f in features if abs(f) > 1e-8)


def main():
    print("=" * 60)
    print("Feature Reprocessing (fixed parsers + 600s window)")
    print("=" * 60)

    # Load existing dataset for price/label matching
    print("\nLoading existing price data...")
    hist_path = os.path.join(RESULTS_DIR, "historical_dataset_v2.json")
    if not os.path.exists(hist_path):
        hist_path = os.path.join(RESULTS_DIR, "historical_dataset.json")
    real_path = os.path.join(RESULTS_DIR, "real_dataset.json")

    price_lookup = {}
    for path in [hist_path, real_path]:
        if os.path.exists(path):
            lookup = build_price_lookup(path)
            for asset, entries in lookup.items():
                if asset not in price_lookup:
                    price_lookup[asset] = []
                price_lookup[asset].extend(entries)
            print(f"  Added from {os.path.basename(path)}")

    # De-dup and sort
    for asset in price_lookup:
        seen = set()
        deduped = []
        for e in price_lookup[asset]:
            key = (e["epoch"], e["price"])
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        deduped.sort(key=lambda x: x["epoch"])
        price_lookup[asset] = deduped
        print(f"    {asset}: {len(deduped)} price entries")

    # Load and group signals with 600s window
    print(f"\nLoading signals from {SIGNALS_FILE}...")
    signals_by_asset = load_signals(SIGNALS_FILE)
    assets = sorted(signals_by_asset.keys())
    total_signals = sum(len(v) for v in signals_by_asset.values())
    print(f"  {total_signals:,} signals across {len(assets)} assets")

    print("\nGrouping with 600s window...")
    snapshots_by_asset = {}
    for asset in assets:
        snaps = group_into_snapshots(signals_by_asset[asset], window_seconds=600)
        snapshots_by_asset[asset] = snaps
        agents_hist = defaultdict(int)
        for snap in snaps:
            n = len(snap.get("agents", {}))
            agents_hist[n] += 1
        print(f"  {asset}: {len(snaps)} snapshots | "
              f"agent coverage: {dict(sorted(agents_hist.items()))}")

    # Process each asset
    print("\nReprocessing features...")
    all_samples = []
    stats = defaultdict(int)

    for asset in assets:
        builder = QuantumVectorBuilder(window_size=252)
        snaps = snapshots_by_asset[asset]
        asset_count = 0
        asset_matched = 0

        for snap in snaps:
            captured_at = parse_timestamp(snap["captured_at"])
            agents = snap.get("agents", {})

            technical = agents.get("technical") or {}
            macro = agents.get("macro") or {}
            news = agents.get("news") or {}
            fundamental = agents.get("fundamental") or {}
            risk_sentiment = agents.get("risk_sentiment") or {}
            commodity = agents.get("commodity") or {}

            # Find matching price/label from existing dataset
            price_info = find_closest_price(price_lookup, asset, captured_at)
            if price_info is None:
                stats["no_price_match"] += 1
                continue

            try:
                vector = builder.build(
                    technical=technical,
                    macro=macro,
                    news=news,
                    fundamental=fundamental,
                    risk_sentiment=risk_sentiment,
                    commodity=commodity,
                    asset=asset,
                    timestamp=snap["captured_at"],
                )

                features_14q = vector.to_array()
                features_8q = vector.to_reduced_8q()

                # Note: FAST_FEATURE_SLOT overwrites are skipped here since
                # we don't have price_bars. Slots 6 and 12 will come from
                # the builder directly (yield_slope and dxy_momentum).

                alive = count_alive(features_14q)
                asset_count += 1
                asset_matched += 1

                all_samples.append({
                    "features_8q": features_8q,
                    "features_14q": features_14q,
                    "label": price_info["label"],
                    "timestamp": snap["captured_at"],
                    "price": price_info["price"],
                    "fwd_return": price_info["fwd_return"],
                    "asset": asset,
                    "agents_active": snap.get("agents_active", 0),
                    "source": "reprocessed_v3",
                    "alive_features": alive,
                })

            except Exception as e:
                stats["build_error"] += 1
                continue

        print(f"  {asset}: {asset_matched}/{len(snaps)} matched → {asset_count} samples")
        stats["total_matched"] += asset_matched

    print(f"\nTotal: {len(all_samples)} samples")
    print(f"Stats: {dict(stats)}")

    # Analyze feature alive counts
    alive_counts = [s["alive_features"] for s in all_samples]
    print(f"\nFeature alive distribution:")
    from collections import Counter
    alive_dist = Counter(alive_counts)
    for n_alive in sorted(alive_dist.keys()):
        pct = alive_dist[n_alive] / len(all_samples) * 100
        bar = "█" * int(pct / 2)
        print(f"  {n_alive:2d}/14: {alive_dist[n_alive]:6d} ({pct:5.1f}%) {bar}")

    # Show per-date alive stats for Mar 13+
    print("\nPer-day average features alive (Mar 13+):")
    by_date = defaultdict(list)
    for s in all_samples:
        date = s["timestamp"][:10]
        if date >= "2026-03-13":
            by_date[date].append(s["alive_features"])

    for date in sorted(by_date.keys())[:10]:
        vals = by_date[date]
        avg = sum(vals) / len(vals)
        n_pass_10 = sum(1 for v in vals if v >= 10)
        print(f"  {date}: avg={avg:.1f}/14, pass≥10: {n_pass_10}/{len(vals)}")

    # Show a sample from Mar 13
    print("\nSample from Mar 13 (first EURUSD):")
    FEATURE_NAMES = [
        "mom_comp", "mom_acc", "vol_20d", "vol_reg", "atr",
        "carry", "yield_slope", "event_risk", "net_sent",
        "sent_mom", "cot_pos", "cot_ext", "dxy_mom", "risk_sent"
    ]
    for s in all_samples:
        if s["asset"] == "EURUSD" and s["timestamp"].startswith("2026-03-13"):
            for i, (name, val) in enumerate(zip(FEATURE_NAMES, s["features_14q"])):
                status = "ALIVE" if abs(val) > 1e-8 else "DEAD"
                print(f"  [{i:2d}] {name:15s} = {val:+.6f}  {status}")
            print(f"  Alive: {s['alive_features']}/14")
            break

    # Save
    # Split: simple 80/20
    np.random.seed(42)
    indices = np.arange(len(all_samples))
    np.random.shuffle(indices)
    n_train = int(len(all_samples) * 0.8)

    train = [all_samples[i] for i in indices[:n_train]]
    val = [all_samples[i] for i in indices[n_train:]]

    output = {
        "metadata": {
            "version": "v3_reprocessed",
            "fixes": [
                "600s grouping window (was 60s)",
                "technical parser handles lowercase format",
                "macro/news/fundamental parsers handle new format",
            ],
            "group_window": 600,
        },
        "train": train,
        "val": val,
    }

    out_path = os.path.join(RESULTS_DIR, "historical_dataset_v3.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"  Train: {len(train)}, Val: {len(val)}")


if __name__ == "__main__":
    main()
