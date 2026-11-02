#!/usr/bin/env python3 -u
"""
Snapshot-to-Dataset Bridge
============================
Converts captured live agent snapshots into the labeled training dataset
format expected by Phase 1D training pipeline.

This is the critical bridge between:
  INPUT:  phase0/snapshots/{ASSET}/*.json  (live Redis captures)
  OUTPUT: phase0/results/real_dataset.json  (List[Dict] with features_8q + labels)

The output format matches exactly what phase1d_train.py expects:
  [{"features_8q": [...], "features_14q": [...], "label": 0|1,
    "timestamp": "...", "price": ..., "fwd_return": ..., "asset": "..."}]

Label target: sign(price[t + forward_hours] - price[t])
Default forward_hours = 4 (matching Phase 1D best setup).

Usage:
  python scripts/snapshot_to_dataset.py                          # All 6/6 assets, default 4h
  python scripts/snapshot_to_dataset.py --assets GBPUSD USDJPY   # Specific assets
  python scripts/snapshot_to_dataset.py --forward-hours 1        # 1h horizon
  python scripts/snapshot_to_dataset.py --min-agents 4           # Require ≥4/6 agents
  python scripts/snapshot_to_dataset.py --inspect                # Print stats, no save
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. Install: pip install numpy")
    sys.exit(1)

from prototypes.other.quantum_signal_gen1_infra.builder import QuantumVectorBuilder
from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector

PHASE0_DIR = Path(__file__).parent.parent
SNAPSHOTS_DIR = PHASE0_DIR / "snapshots"
RESULTS_DIR = PHASE0_DIR / "results"

# Assets with full 6/6 agent coverage
FULL_COVERAGE_ASSETS = [
    "GBPUSD", "USDJPY", "USDCHF", "EURUSD", "AUDUSD", "USDCAD",  # FX
    "SPY", "NVDA", "AAPL", "MSFT", "TSLA",                         # Equities
]

# No partial assets remaining — all configured for 6/6
PARTIAL_ASSETS = []


# ── Noise Filtering (Fix 3) ─────────────────────────────────────────

# Pip size per asset — used to convert forward returns to pip-scale moves
PIP_SIZE = {
    "GBPUSD": 0.0001, "EURUSD": 0.0001, "AUDUSD": 0.0001,
    "USDCHF": 0.0001, "USDCAD": 0.0001,
    "USDJPY": 0.01,
}
# Equities don't have pips — use a percentage threshold instead
# 10 pips ≈ 0.10% for equities (calibrated to typical bid-ask spreads)
EQUITY_MIN_MOVE_PCT = 0.001  # 0.1% per 10 "pips"


def move_exceeds_threshold(fwd_return: float, asset: str,
                           min_pips: int, current_price: float) -> bool:
    """
    Check if the absolute forward move exceeds the noise threshold.

    For FX: converts return to pips using the pair's pip size.
    For equities: uses a percentage threshold scaled by min_pips.

    Args:
        fwd_return: (forward_price - current_price) / current_price
        min_pips: minimum move in pips (FX) or pip-equivalents (equities)
        current_price: price at snapshot time (needed for FX pip conversion)

    Returns:
        True if move is large enough to be signal, False if likely noise.
    """
    pip = PIP_SIZE.get(asset)
    if pip is not None:
        # FX: |return| * price / pip_size = move in pips
        move_pips = abs(fwd_return) * current_price / pip
        return move_pips >= min_pips
    else:
        # Equity: use percentage threshold (0.1% per 10 pips)
        min_pct = EQUITY_MIN_MOVE_PCT * (min_pips / 10.0)
        return abs(fwd_return) >= min_pct


# ── Fast Features from Yahoo Bars (Fix 4) ────────────────────────────
# These features update on the 4h timescale, matching our prediction horizon.
# Computed during conversion from Yahoo hourly bars — no snapshot changes needed.
# Injected into 14Q slots 6 (yield_curve_slope, 100% zeros) and 12 (dxy_momentum, 100% zeros).

def compute_price_momentum_4h(price_bars: List[Dict],
                               snapshot_time: datetime,
                               max_gap_hours: float = 2.0) -> float:
    """
    4-hour price momentum: tanh-scaled return over the last 4 hours.

    Uses Yahoo hourly bars to find prices at t and t-4h.
    Returns value in [-1, 1] where ±0.5% move ≈ ±0.46.
    Returns 0.0 (neutral) if data unavailable.
    """
    current_price = find_price_at(price_bars, snapshot_time, max_gap_hours)
    if current_price is None:
        return 0.0

    lookback_time = snapshot_time - timedelta(hours=4)
    past_price = find_price_at(price_bars, lookback_time, max_gap_hours)
    if past_price is None or past_price == 0:
        return 0.0

    raw_return = (current_price - past_price) / past_price
    # tanh scaling: 0.5% move → ~0.46, 1% → ~0.76, 2% → ~0.96
    return max(-1.0, min(1.0, math.tanh(raw_return * 100)))


def compute_intraday_vol(price_bars: List[Dict],
                          snapshot_time: datetime,
                          lookback_hours: int = 8,
                          max_gap_hours: float = 2.0) -> float:
    """
    Intraday volatility: std of recent hourly returns.

    Uses the last `lookback_hours` hourly bars before snapshot_time.
    Normalized to [0, 1] where 0.5% hourly std = 1.0.
    Returns 0.5 (neutral) if insufficient data.
    """
    if not price_bars or len(price_bars) < 2:
        return 0.5

    # Collect bars in the lookback window
    window_start = snapshot_time - timedelta(hours=lookback_hours)
    window_bars = [b for b in price_bars
                   if window_start <= b["timestamp"] <= snapshot_time]

    if len(window_bars) < 3:
        return 0.5

    # Compute hourly returns
    closes = [b["close"] for b in window_bars]
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

    if len(returns) < 2:
        return 0.5

    # Standard deviation of returns
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std = var ** 0.5

    # Normalize: 0.5% hourly std → 1.0
    return min(1.0, std / 0.005)


# 14Q slot indices for injecting fast features (replacing dead slots)
FAST_FEATURE_SLOT_PRICE_MOM = 6   # was yield_curve_slope (100% zeros)
FAST_FEATURE_SLOT_INTRADAY_VOL = 12  # was dxy_momentum (100% zeros)


# ── Price Data ────────────────────────────────────────────────────────

def fetch_price_series(symbol: str, period: str = "60d",
                       interval: str = "1h") -> Optional[List[Dict]]:
    """
    Fetch OHLCV price data from Yahoo Finance for label computation.
    Returns list of {timestamp: datetime, open, high, low, close, volume}.
    Sorted by timestamp ascending.
    """
    import urllib.request

    yahoo_symbol = symbol
    if symbol in ("GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"):
        yahoo_symbol = f"{symbol[:3]}{symbol[3:]}=X"

    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
           f"?range={period}&interval={interval}&includePrePost=false")

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  WARNING: Failed to fetch {symbol}: {e}")
        return None

    result = data.get("chart", {}).get("result", [])
    if not result:
        print(f"  WARNING: No data for {symbol}")
        return None

    r = result[0]
    timestamps = r.get("timestamp", [])
    quote = r.get("indicators", {}).get("quote", [{}])[0]

    bars = []
    for i, ts in enumerate(timestamps):
        c = quote.get("close", [None])[i]
        o = quote.get("open", [None])[i]
        h = quote.get("high", [None])[i]
        lo = quote.get("low", [None])[i]
        v = quote.get("volume", [0])[i]

        if c is None or o is None:
            continue

        bars.append({
            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
            "open": float(o),
            "high": float(h) if h else float(o),
            "low": float(lo) if lo else float(o),
            "close": float(c),
            "volume": int(v) if v else 0,
        })

    bars.sort(key=lambda x: x["timestamp"])
    return bars


def find_price_at(bars: List[Dict], target_time: datetime,
                  max_gap_hours: float = 2.0) -> Optional[float]:
    """
    Find the closest price bar to target_time.
    Returns the close price if within max_gap_hours, else None.
    Uses binary search for efficiency.
    """
    if not bars:
        return None

    # Binary search for closest bar
    lo, hi = 0, len(bars) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if bars[mid]["timestamp"] < target_time:
            lo = mid + 1
        else:
            hi = mid

    # Check neighbors for closest
    best_idx = lo
    best_gap = abs((bars[lo]["timestamp"] - target_time).total_seconds())

    if lo > 0:
        gap = abs((bars[lo - 1]["timestamp"] - target_time).total_seconds())
        if gap < best_gap:
            best_idx = lo - 1
            best_gap = gap

    if lo < len(bars) - 1:
        gap = abs((bars[lo + 1]["timestamp"] - target_time).total_seconds())
        if gap < best_gap:
            best_idx = lo + 1
            best_gap = gap

    if best_gap > max_gap_hours * 3600:
        return None

    return bars[best_idx]["close"]


def find_forward_price(bars: List[Dict], snapshot_time: datetime,
                       forward_hours: int = 4,
                       max_gap_hours: float = 2.0) -> Optional[Tuple[float, float, float]]:
    """
    Find price at snapshot_time and at snapshot_time + forward_hours.
    Returns (current_price, forward_price, forward_return) or None.
    """
    current_price = find_price_at(bars, snapshot_time, max_gap_hours=max_gap_hours)
    if current_price is None:
        return None

    target_time = snapshot_time + timedelta(hours=forward_hours)
    forward_price = find_price_at(bars, target_time, max_gap_hours=max_gap_hours)
    if forward_price is None:
        return None

    fwd_return = (forward_price - current_price) / max(current_price, 1e-10)
    return current_price, forward_price, fwd_return


# ── Snapshot Loading ──────────────────────────────────────────────────

def load_snapshots(asset: str, min_agents: int = 4) -> List[Dict]:
    """
    Load all snapshots for an asset from disk.
    Filters by minimum active agent count.
    Returns sorted by captured_at ascending.
    """
    snap_dir = SNAPSHOTS_DIR / asset
    if not snap_dir.exists():
        print(f"  WARNING: No snapshot directory for {asset}")
        return []

    snapshots = []
    files = sorted(snap_dir.glob("*.json"))

    for f in files:
        try:
            with open(f) as fh:
                snap = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            continue

        if snap.get("agents_active", 0) < min_agents:
            continue

        snapshots.append(snap)

    # Sort by capture time
    snapshots.sort(key=lambda s: s.get("captured_at", ""))
    return snapshots


def extract_agent_dicts(snap: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Extract the 6 agent signal dictionaries from a snapshot.
    Returns (technical, macro, news, fundamental, risk_sentiment, commodity).
    Missing agents get empty dicts (builder handles defaults).
    """
    agents = snap.get("agents", {})

    technical = agents.get("technical") or {}
    macro = agents.get("macro") or {}
    news = agents.get("news") or {}
    fundamental = agents.get("fundamental") or {}
    risk_sentiment = agents.get("risk_sentiment") or {}
    commodity = agents.get("commodity") or {}

    return technical, macro, news, fundamental, risk_sentiment, commodity


# ── Deduplication ────────────────────────────────────────────────────

def deduplicate_snapshots(snapshots: List[Dict],
                          min_interval_seconds: int = 300) -> List[Dict]:
    """
    Remove near-duplicate snapshots captured within min_interval of each other.
    Keeps the first snapshot in each time window.

    Default: keep one snapshot per 5-minute window (300s).
    With 60s polling, this reduces 60 snaps/hour to 12 snaps/hour,
    each with meaningfully different agent data.
    """
    if not snapshots:
        return []

    deduped = [snapshots[0]]
    last_time = datetime.fromisoformat(snapshots[0]["captured_at"])

    for snap in snapshots[1:]:
        snap_time = datetime.fromisoformat(snap["captured_at"])
        gap = (snap_time - last_time).total_seconds()

        if gap >= min_interval_seconds:
            deduped.append(snap)
            last_time = snap_time

    return deduped


# ── Main Conversion ──────────────────────────────────────────────────

def convert_asset(asset: str, price_bars: List[Dict],
                  snapshots: List[Dict], forward_hours: int = 4,
                  dedup_interval: int = 300,
                  min_move_pips: int = 0) -> Tuple[List[Dict], Dict]:
    """
    Convert snapshots for one asset into labeled training rows.

    Args:
        min_move_pips: Minimum forward move in pips to include a sample.
                       Samples below this threshold are noise (bid-ask spread).
                       0 = disabled (keep all samples). Default: 0.

    Returns:
        (dataset_rows, stats_dict)
    """
    builder = QuantumVectorBuilder(window_size=252)

    # Deduplicate
    snapshots = deduplicate_snapshots(snapshots, min_interval_seconds=dedup_interval)

    stats = {
        "asset": asset,
        "total_snapshots": len(snapshots),
        "valid_vectors": 0,
        "labeled": 0,
        "skipped_no_price": 0,
        "skipped_build_error": 0,
        "skipped_noise": 0,
        "label_balance": 0.0,
    }

    dataset = []

    for snap in snapshots:
        captured_at = datetime.fromisoformat(snap["captured_at"])
        technical, macro, news, fundamental, risk_sentiment, commodity = extract_agent_dicts(snap)

        # Build quantum vector from real agent data
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

            features_8q = vector.to_reduced_8q()
            features_14q = vector.to_array()

            # Fix 4: Inject fast-update features into dead 14Q slots
            # Slot 6 (yield_curve_slope, was 100% zeros) → price_momentum_4h
            # Slot 12 (dxy_momentum, was 100% zeros) → intraday_vol
            features_14q[FAST_FEATURE_SLOT_PRICE_MOM] = compute_price_momentum_4h(
                price_bars, captured_at)
            features_14q[FAST_FEATURE_SLOT_INTRADAY_VOL] = compute_intraday_vol(
                price_bars, captured_at)

            stats["valid_vectors"] += 1

        except Exception as e:
            stats["skipped_build_error"] += 1
            continue

        # Find price and forward label
        price_info = find_forward_price(
            price_bars, captured_at,
            forward_hours=forward_hours,
            max_gap_hours=2.0,
        )

        if price_info is None:
            stats["skipped_no_price"] += 1
            continue

        current_price, forward_price, fwd_return = price_info

        # Fix 3: Filter samples where forward move is below noise threshold
        if min_move_pips > 0:
            if not move_exceeds_threshold(fwd_return, asset,
                                          min_move_pips, current_price):
                stats["skipped_noise"] += 1
                continue

        label = 1 if fwd_return > 0 else 0

        dataset.append({
            "features_8q": features_8q,
            "features_14q": features_14q,
            "label": label,
            "timestamp": snap["captured_at"],
            "price": current_price,
            "fwd_return": fwd_return,
            "asset": asset,
            "agents_active": snap.get("agents_active", 0),
        })

        stats["labeled"] += 1

    if stats["labeled"] > 0:
        positives = sum(1 for d in dataset if d["label"] == 1)
        stats["label_balance"] = positives / stats["labeled"]

    return dataset, stats


# ── Dataset Splitting ─────────────────────────────────────────────────

def split_by_time(dataset: List[Dict], train_ratio: float = 0.75) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset by time ordering (no shuffling — prevents lookahead).
    First train_ratio% is train, rest is validation.
    DEPRECATED: Use split_by_time_blocks() instead — this has 97.9% label overlap.
    """
    if not dataset:
        return [], []

    # Already sorted by timestamp from conversion
    dataset.sort(key=lambda d: d["timestamp"])
    split_idx = int(len(dataset) * train_ratio)

    return dataset[:split_idx], dataset[split_idx:]


def split_by_time_blocks(dataset: List[Dict],
                         train_ratio: float = 0.75,
                         block_hours: int = 4,
                         gap_hours: int = 4) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into non-overlapping temporal blocks to eliminate label leakage.

    With 5-min sampling and 4h forward returns, consecutive samples share almost
    identical labels (97.9% overlap). This function groups samples into non-overlapping
    time blocks and splits BLOCKS, not individual samples.

    A gap of gap_hours between the last train block and first val block ensures
    zero label contamination between sets.

    Args:
        dataset: list of sample dicts, each with "timestamp" field
        train_ratio: fraction of BLOCKS (not samples) for training
        block_hours: hours per block (should match forward prediction horizon)
        gap_hours: gap between last train block and first val block (= forward horizon)

    Returns:
        (train_samples, val_samples) — each sorted by timestamp
    """
    if not dataset:
        return [], []

    dataset.sort(key=lambda d: d["timestamp"])

    # Assign each sample to a block ID based on its timestamp
    base_time = datetime.fromisoformat(dataset[0]["timestamp"])
    block_seconds = block_hours * 3600

    blocks = defaultdict(list)
    for sample in dataset:
        t = datetime.fromisoformat(sample["timestamp"])
        delta = (t - base_time).total_seconds()
        block_id = int(delta // block_seconds)
        blocks[block_id].append(sample)

    # Sort block IDs chronologically
    sorted_block_ids = sorted(blocks.keys())
    n_blocks = len(sorted_block_ids)

    if n_blocks < 2:
        print(f"  WARNING: Only {n_blocks} block(s) — cannot split. All data goes to train.")
        return dataset, []

    n_train_blocks = max(1, int(n_blocks * train_ratio))

    train_block_ids = set(sorted_block_ids[:n_train_blocks])

    # Gap: skip blocks that fall within gap_hours of last train block
    last_train_block = sorted_block_ids[n_train_blocks - 1]
    gap_blocks = max(1, int(gap_hours / block_hours))  # at least 1 block gap

    val_block_ids = set()
    for bid in sorted_block_ids[n_train_blocks:]:
        if bid > last_train_block + gap_blocks:
            val_block_ids.add(bid)

    train = [s for bid in sorted_block_ids if bid in train_block_ids
             for s in blocks[bid]]
    val = [s for bid in sorted_block_ids if bid in val_block_ids
           for s in blocks[bid]]

    # Report statistics
    n_gap_samples = sum(len(blocks[bid]) for bid in sorted_block_ids[n_train_blocks:]
                        if bid not in val_block_ids)
    samples_per_block = [len(blocks[bid]) for bid in sorted_block_ids]
    avg_per_block = sum(samples_per_block) / len(samples_per_block) if samples_per_block else 0

    print(f"  Block split: {n_blocks} total blocks ({block_hours}h each), "
          f"avg {avg_per_block:.1f} samples/block")
    print(f"    Train: {len(train_block_ids)} blocks → {len(train)} samples")
    print(f"    Gap:   {gap_blocks} block(s) dropped → {n_gap_samples} samples removed")
    print(f"    Val:   {len(val_block_ids)} blocks → {len(val)} samples")

    if not val:
        print(f"  WARNING: No val blocks after gap! Need more data or smaller gap.")
        print(f"    Total blocks: {n_blocks}, train: {n_train_blocks}, "
              f"gap needs: {gap_blocks}, available after train: {n_blocks - n_train_blocks}")

    return train, val


# ── Inspection ────────────────────────────────────────────────────────

def inspect_dataset(dataset: List[Dict], label: str = "Dataset"):
    """Print detailed statistics about the dataset."""
    if not dataset:
        print(f"\n{label}: EMPTY")
        return

    features = np.array([d["features_8q"] for d in dataset])
    labels = np.array([d["label"] for d in dataset])
    assets = [d["asset"] for d in dataset]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Label balance: {labels.mean():.1%} positive")
    print(f"  Time range: {dataset[0]['timestamp'][:19]} → {dataset[-1]['timestamp'][:19]}")

    # Per-asset breakdown
    asset_counts = defaultdict(int)
    for a in assets:
        asset_counts[a] += 1
    print(f"  Assets: {dict(asset_counts)}")

    # Feature statistics (8q)
    feature_names = [
        "momentum_composite", "realized_vol_20d", "momentum_acceleration",
        "atr_normalized", "event_risk_score", "net_sentiment",
        "sentiment_momentum", "risk_sentiment",
    ]
    print(f"\n  Feature Statistics (8Q):")
    print(f"  {'Feature':<25s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Unique>5':>9s}")
    print(f"  {'─'*70}")

    dead_features = []
    for i, name in enumerate(feature_names):
        col = features[:, i]
        n_unique = len(np.unique(np.round(col, 4)))
        is_dead = n_unique <= 3 or col.std() < 0.001
        marker = " ⚠ DEAD" if is_dead else ""
        print(f"  {name:<25s} {col.mean():>8.4f} {col.std():>8.4f} "
              f"{col.min():>8.4f} {col.max():>8.4f} {n_unique:>9d}{marker}")
        if is_dead:
            dead_features.append(name)

    if dead_features:
        print(f"\n  WARNING: {len(dead_features)} dead/near-constant features: {dead_features}")

    # Forward return distribution
    fwd_returns = np.array([d["fwd_return"] for d in dataset])
    print(f"\n  Forward returns: mean={fwd_returns.mean():.6f}, "
          f"std={fwd_returns.std():.6f}, "
          f"min={fwd_returns.min():.6f}, max={fwd_returns.max():.6f}")


# ── Feature comparison ────────────────────────────────────────────────

def compare_proxy_vs_real(real_dataset: List[Dict]):
    """
    Print comparison notes between proxy dataset characteristics
    and real dataset characteristics.
    """
    if not real_dataset:
        return

    features = np.array([d["features_8q"] for d in real_dataset])
    names = [
        "q0_momentum", "q1_vol", "q2_mom_accel",
        "q3_atr", "q4_event", "q5_sentiment",
        "q6_sent_mom", "q7_risk",
    ]

    print(f"\n{'='*60}")
    print(f"  PROXY vs REAL Feature Analysis")
    print(f"{'='*60}")
    print(f"  Feature       Proxy Source              Real Source             Std")
    print(f"  {'─'*70}")

    proxy_notes = [
        ("q0 momentum",   "RSI+MACD+BB (real)",      "RSI+MACD+BB (real)"),
        ("q1 vol",        "ATR approximation",       "ATR approximation"),
        ("q2 mom_accel",  "Momentum delta",           "Momentum delta"),
        ("q3 atr",        "ATR normalized",            "ATR normalized"),
        ("q4 event",      "Hardcoded 24h DEAD",       "Real calendar hours"),
        ("q5 sentiment",  "Synthetic from RSI PROXY", "Real RSS bull/bear counts"),
        ("q6 sent_mom",   "Sentiment 3d delta",        "Sentiment 3d delta"),
        ("q7 risk",       "15+vol*5000 PROXY",        "Real VIX level"),
    ]

    for i, (name, proxy_src, real_src) in enumerate(proxy_notes):
        std = features[:, i].std()
        marker = "✓ REAL" if i in [4, 5, 6, 7] else "same"
        print(f"  {name:<15s} {proxy_src:<25s} {real_src:<25s} {std:.4f} {marker}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert captured snapshots to Phase 1D training dataset"
    )
    parser.add_argument("--assets", nargs="+", default=FULL_COVERAGE_ASSETS,
                        help=f"Assets to include (default: {FULL_COVERAGE_ASSETS})")
    parser.add_argument("--forward-hours", type=int, default=4,
                        help="Forward return horizon for labels (default: 4)")
    parser.add_argument("--min-agents", type=int, default=4,
                        help="Minimum active agents per snapshot (default: 4)")
    parser.add_argument("--dedup-interval", type=int, default=300,
                        help="Min seconds between snapshots after dedup (default: 300)")
    parser.add_argument("--train-ratio", type=float, default=0.75,
                        help="Train/val split ratio (default: 0.75)")
    parser.add_argument("--price-period", default="60d",
                        help="Yahoo price data period (default: 60d)")
    parser.add_argument("--min-move-pips", type=int, default=10,
                        help="Min forward move in pips to include sample (default: 10, 0=disabled)")
    parser.add_argument("--no-block-split", action="store_true",
                        help="Use old simple time split instead of block split (not recommended)")
    parser.add_argument("--inspect", action="store_true",
                        help="Print dataset stats only, don't save")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: results/real_dataset.json)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else RESULTS_DIR / "real_dataset.json"

    print("=" * 60)
    print("  SNAPSHOT → DATASET CONVERSION")
    print("=" * 60)
    print(f"  Assets:          {args.assets}")
    print(f"  Forward horizon: {args.forward_hours}h")
    print(f"  Min agents:      {args.min_agents}/6")
    print(f"  Dedup interval:  {args.dedup_interval}s")
    print(f"  Min move:        {args.min_move_pips} pips {'(disabled)' if args.min_move_pips == 0 else ''}")
    print(f"  Train ratio:     {args.train_ratio}")
    print(f"  Snapshot dir:    {SNAPSHOTS_DIR}")
    print(f"  Output:          {output_path}")
    print()

    all_data = []
    all_stats = []

    for asset in args.assets:
        print(f"─── {asset} ───")

        # Load snapshots
        snapshots = load_snapshots(asset, min_agents=args.min_agents)
        if not snapshots:
            print(f"  No valid snapshots for {asset}, skipping.")
            continue
        print(f"  Loaded {len(snapshots)} snapshots (≥{args.min_agents} agents)")

        # Fetch price data
        print(f"  Fetching price data from Yahoo ({args.price_period})...")
        price_bars = fetch_price_series(asset, period=args.price_period, interval="1h")
        if not price_bars:
            print(f"  No price data for {asset}, skipping.")
            continue
        print(f"  Got {len(price_bars)} price bars "
              f"({price_bars[0]['timestamp'].strftime('%Y-%m-%d')} → "
              f"{price_bars[-1]['timestamp'].strftime('%Y-%m-%d')})")

        # Convert
        dataset, stats = convert_asset(
            asset=asset,
            price_bars=price_bars,
            snapshots=snapshots,
            forward_hours=args.forward_hours,
            dedup_interval=args.dedup_interval,
            min_move_pips=args.min_move_pips,
        )

        print(f"  Vectors built: {stats['valid_vectors']}")
        print(f"  Labeled:       {stats['labeled']}")
        print(f"  No price:      {stats['skipped_no_price']}")
        print(f"  Build errors:  {stats['skipped_build_error']}")
        if stats.get("skipped_noise", 0) > 0:
            noise_pct = stats["skipped_noise"] / (stats["labeled"] + stats["skipped_noise"]) * 100
            print(f"  Noise filter:  {stats['skipped_noise']} removed ({noise_pct:.0f}%)")
        if stats["labeled"] > 0:
            print(f"  Label balance: {stats['label_balance']:.1%} positive")

        all_data.extend(dataset)
        all_stats.append(stats)

    if not all_data:
        print("\nERROR: No labeled data produced. Possible causes:")
        print("  - Snapshots were captured outside market hours (no price bars match)")
        print("  - Not enough capture time (need forward_hours beyond last snapshot)")
        print("  - Yahoo price data doesn't cover snapshot time range")
        print("\nTIP: If you just started capturing, you need to wait at least")
        print(f"     {args.forward_hours} hours beyond the first snapshot before labels exist.")
        print("     Snapshots captured on weekends won't have FX price data.")
        sys.exit(1)

    # Sort all data by time
    all_data.sort(key=lambda d: d["timestamp"])

    # Inspect
    inspect_dataset(all_data, "FULL DATASET (Real Agent Features)")
    compare_proxy_vs_real(all_data)

    # Split — use temporal block splitting by default (Fix 1: eliminates label leakage)
    if args.no_block_split:
        print("\n  Using LEGACY simple time split (label leakage possible!)")
        train_data, val_data = split_by_time(all_data, train_ratio=args.train_ratio)
    else:
        print(f"\n  Using temporal block split ({args.forward_hours}h blocks, "
              f"{args.forward_hours}h gap):")
        train_data, val_data = split_by_time_blocks(
            all_data,
            train_ratio=args.train_ratio,
            block_hours=args.forward_hours,
            gap_hours=args.forward_hours,
        )
    print(f"\n  Final split: {len(train_data)} train / {len(val_data)} val")

    inspect_dataset(train_data, "TRAIN SET")
    inspect_dataset(val_data, "VALIDATION SET")

    if args.inspect:
        print("\n[inspect mode — not saving]")
        return

    # Save
    output = {
        "metadata": {
            "created": datetime.now(timezone.utc).isoformat(),
            "forward_hours": args.forward_hours,
            "min_agents": args.min_agents,
            "dedup_interval_s": args.dedup_interval,
            "min_move_pips": args.min_move_pips,
            "assets": args.assets,
            "total_samples": len(all_data),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "split_method": "simple_time" if args.no_block_split else "temporal_blocks",
            "block_hours": None if args.no_block_split else args.forward_hours,
            "gap_hours": None if args.no_block_split else args.forward_hours,
            "source": "live_agent_snapshots",
            "conversion_stats": all_stats,
        },
        "train": train_data,
        "val": val_data,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DATASET READY")
    print(f"{'='*60}")
    print(f"  Total: {len(all_data)} samples ({len(train_data)} train / {len(val_data)} val)")
    print(f"  Assets: {args.assets}")
    print(f"  Horizon: {args.forward_hours}h forward return")
    print(f"  Source: REAL agent snapshots (not Yahoo-reconstructed proxies)")
    print()
    print(f"  Next step — train Phase 1D on real data:")
    print(f"    python scripts/phase1d_real_train.py --dataset {output_path}")
    print()


if __name__ == "__main__":
    main()
