#!/usr/bin/env python3 -u
"""
Signals-to-Dataset Converter
=============================
Converts exported PostgreSQL signal JSONL files into the labeled training
dataset format expected by Phase 1D training pipeline.

This is the bridge between:
  INPUT:  signals_export.jsonl  (PostgreSQL COPY output, one JSON per line)
  OUTPUT: phase0/results/real_dataset.json  (same format as snapshot_to_dataset.py)

The output format matches exactly what phase1d_train.py expects:
  [{"features_8q": [...], "features_14q": [...], "label": 0|1,
    "timestamp": "...", "price": ..., "fwd_return": ..., "asset": "..."}]

Signal reasoning strings are parsed to extract structured features
(RSI, MACD, BB, ATR, carry, events) and fed through the same
QuantumVectorBuilder used for live snapshots.

Usage:
  python scripts/signals_to_dataset.py                                    # Default: all assets
  python scripts/signals_to_dataset.py --signals-file /path/to/export.jsonl
  python scripts/signals_to_dataset.py --assets GBPUSD USDJPY            # Specific assets
  python scripts/signals_to_dataset.py --forward-hours 4 --min-move-pips 10
  python scripts/signals_to_dataset.py --inspect                          # Print stats, no save
  python scripts/signals_to_dataset.py --output results/historical_dataset.json
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dateutil.parser import isoparse as _dateutil_isoparse


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp robustly (handles variable microsecond precision)."""
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return _dateutil_isoparse(ts)


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
RESULTS_DIR = PHASE0_DIR / "results"

# Default signals file location (relative to the project root)
DEFAULT_SIGNALS_FILE = PHASE0_DIR.parent / "signals_export.jsonl"

# ── Noise Filtering (Fix 3) ─────────────────────────────────────────
PIP_SIZE = {
    "GBPUSD": 0.0001, "EURUSD": 0.0001, "AUDUSD": 0.0001,
    "USDCHF": 0.0001, "USDCAD": 0.0001,
    "USDJPY": 0.01,
}
EQUITY_MIN_MOVE_PCT = 0.001


def move_exceeds_threshold(fwd_return: float, asset: str,
                           min_pips: int, current_price: float) -> bool:
    """Check if the absolute forward move exceeds the noise threshold."""
    pip = PIP_SIZE.get(asset)
    if pip is not None:
        move_pips = abs(fwd_return) * current_price / pip
        return move_pips >= min_pips
    else:
        min_pct = EQUITY_MIN_MOVE_PCT * (min_pips / 10.0)
        return abs(fwd_return) >= min_pct


# ── Fast Features (Fix 4) ───────────────────────────────────────────
FAST_FEATURE_SLOT_PRICE_MOM = 6
FAST_FEATURE_SLOT_INTRADAY_VOL = 12


def compute_price_momentum_4h(price_bars: List[Dict],
                               snapshot_time: datetime,
                               max_gap_hours: float = 2.0) -> float:
    """4-hour price momentum: tanh-scaled return over the last 4 hours."""
    current_price = find_price_at(price_bars, snapshot_time, max_gap_hours)
    if current_price is None:
        return 0.0
    lookback_time = snapshot_time - timedelta(hours=4)
    past_price = find_price_at(price_bars, lookback_time, max_gap_hours)
    if past_price is None or past_price == 0:
        return 0.0
    raw_return = (current_price - past_price) / past_price
    return max(-1.0, min(1.0, math.tanh(raw_return * 100)))


def compute_intraday_vol(price_bars: List[Dict],
                          snapshot_time: datetime,
                          lookback_hours: int = 8,
                          max_gap_hours: float = 2.0) -> float:
    """Intraday volatility: std of recent hourly returns, normalized [0,1]."""
    if not price_bars or len(price_bars) < 2:
        return 0.5
    window_start = snapshot_time - timedelta(hours=lookback_hours)
    window_bars = [b for b in price_bars
                   if window_start <= b["timestamp"] <= snapshot_time]
    if len(window_bars) < 3:
        return 0.5
    closes = [b["close"] for b in window_bars]
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
    if len(returns) < 2:
        return 0.5
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std = var ** 0.5
    return min(1.0, std / 0.005)


# ── Price Data ───────────────────────────────────────────────────────

def fetch_price_series(symbol: str, period: str = "60d",
                       interval: str = "1h") -> Optional[List[Dict]]:
    """Fetch OHLCV data from Yahoo Finance."""
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
    """Find the closest price bar to target_time within max_gap_hours."""
    if not bars:
        return None

    lo, hi = 0, len(bars) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if bars[mid]["timestamp"] < target_time:
            lo = mid + 1
        else:
            hi = mid

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
    """Find current and forward price. Returns (current, forward, return) or None."""
    current_price = find_price_at(bars, snapshot_time, max_gap_hours=max_gap_hours)
    if current_price is None:
        return None
    target_time = snapshot_time + timedelta(hours=forward_hours)
    forward_price = find_price_at(bars, target_time, max_gap_hours=max_gap_hours)
    if forward_price is None:
        return None
    fwd_return = (forward_price - current_price) / max(current_price, 1e-10)
    return current_price, forward_price, fwd_return


# ═══════════════════════════════════════════════════════════════════════
# REASONING PARSERS — extract structured features from reasoning strings
# ═══════════════════════════════════════════════════════════════════════

def parse_technical_reasoning(reasoning: str) -> Dict:
    """
    Parse a technical agent reasoning string into structured features.

    Handles two formats:
      Early: "RSI=39.79 (oversold), MACD cross=bearish (-0.000249)"
      Later: "RSI=25.11 | MACD=-0.002719 | BB=below_lower | ATR=0.00845 | src=yahoo_finance"
    """
    result = {}

    # RSI — case-insensitive to handle both "RSI=39.79" and "rsi=32.29"
    m = re.search(r'[Rr][Ss][Ii][= ]+([0-9.]+)', reasoning)
    if m:
        result["rsi"] = float(m.group(1))

    # MACD — multiple formats:
    #   Old: "MACD=-0.002719" or "MACD cross=bearish (-0.000249)"
    #   New: "macd_cross=-0.002427"
    m = re.search(r'macd_cross=(-?[0-9.]+)', reasoning)
    if m:
        result["macd_cross"] = float(m.group(1))
    else:
        m = re.search(r'MACD[= ]+(-?[0-9.]+)', reasoning)
        if m:
            result["macd_cross"] = float(m.group(1))
        else:
            m = re.search(r'MACD cross=\w+\s*\((-?[0-9.]+)\)', reasoning)
            if m:
                result["macd_cross"] = float(m.group(1))

    # BB position — handles:
    #   Old: "BB=inside", "BB=below_lower", "BB=above_upper"
    #   New: "bb_ratio=0.0431" (numeric)
    m = re.search(r'bb_ratio=([0-9.]+)', reasoning)
    if m:
        result["bb_position_ratio"] = float(m.group(1))
    else:
        m = re.search(r'BB[= ]+(\w+)', reasoning)
        if m:
            bb_val = m.group(1).lower()
            if bb_val == "above_upper":
                result["bb_position_ratio"] = 0.95
            elif bb_val == "below_lower":
                result["bb_position_ratio"] = 0.05
            elif bb_val == "inside":
                result["bb_position_ratio"] = 0.5
            else:
                try:
                    result["bb_position_ratio"] = float(bb_val)
                except ValueError:
                    result["bb_position_ratio"] = 0.5

    # ATR — case-insensitive for "ATR=0.00845" and "atr=0.00945"
    m = re.search(r'[Aa][Tt][Rr][= ]+([0-9.]+)', reasoning)
    if m:
        result["atr"] = float(m.group(1))

    return result


def parse_macro_reasoning(reasoning: str) -> Dict:
    """
    Parse a macro agent reasoning string into structured features.

    Formats:
      Old: "Carry: EUR=2.9% vs USD=4.25% (diff=-1.35%) -> SELL | 9 high-impact events pending"
      New: "carry=-1.3500 blackout=False next_event=Core CPI m/m"
      Early (technical fallback): "RSI=39.79 (oversold), MACD cross=bearish (-0.000249)"
    """
    result = {}

    # Check if this is actually a technical-format reasoning (early macro signals)
    if "RSI=" in reasoning and "Carry" not in reasoning and "carry=" not in reasoning:
        return parse_technical_reasoning(reasoning)

    # ── New format (Mar 13+): "carry=-1.3500 blackout=False next_event=..." ──
    m = re.search(r'carry=([+-]?[0-9.]+)', reasoning)
    if m:
        result["carry_differential"] = float(m.group(1))

        # next_event presence implies an upcoming event → moderate risk
        if "next_event=" in reasoning:
            event_str = reasoning.split("next_event=")[-1].strip()
            if event_str and event_str.lower() != "none":
                # Has a named event → estimate moderate risk (24h away)
                result["hours_to_next_event"] = 24.0
                result["pending_high_impact_count"] = 1
            else:
                result["hours_to_next_event"] = 168.0
                result["pending_high_impact_count"] = 0

        # blackout=True means we're in a central bank blackout period → higher risk
        if "blackout=True" in reasoning:
            result["hours_to_next_event"] = 6.0
            result["pending_high_impact_count"] = 3

        return result

    # ── Old format: "Carry: EUR=2.9% vs USD=4.25% (diff=-1.35%) -> SELL" ──
    m = re.search(r'diff=([+-]?[0-9.]+)%', reasoning)
    if m:
        result["carry_differential"] = float(m.group(1))

    # Base and quote rates
    m = re.search(r'(\w+)=([0-9.]+)%\s+vs\s+(\w+)=([0-9.]+)%', reasoning)
    if m:
        result["rate_base"] = float(m.group(2))
        result["rate_quote"] = float(m.group(4))

    # High-impact events
    m = re.search(r'(\d+)\s+high-impact\s+events?\s+pending', reasoning)
    if m:
        count = int(m.group(1))
        result["pending_high_impact_count"] = count
        if count > 0:
            result["hours_to_next_event"] = max(1.0, 48.0 / count)
        else:
            result["hours_to_next_event"] = 168.0

    return result


def parse_fundamental_reasoning(reasoning: str) -> Dict:
    """
    Parse a fundamental agent reasoning string.

    Formats:
      Old COT: "COT EUR lev_funds=+100% -> EURUSD BUY"
      New:     "base_cot=13.0 quote_cot=None trend=-0.618 pmi=50.1"
      Early:   "RSI=39.79 (oversold), MACD cross=bearish (-0.000249)"
    """
    result = {}

    # ── New format (Mar 13+): "base_cot=13.0 quote_cot=None trend=-0.618 pmi=50.1" ──
    m = re.search(r'base_cot=([+-]?[0-9.]+)', reasoning)
    if m:
        result["base_cot_net_pct"] = float(m.group(1))

    m = re.search(r'quote_cot=([+-]?[0-9.]+)', reasoning)
    if m:
        result["quote_cot_net_pct"] = float(m.group(1))

    m = re.search(r'trend=([+-]?[0-9.]+)', reasoning)
    if m:
        result["price_trend_pct"] = float(m.group(1))

    m = re.search(r'pmi=([0-9.]+)', reasoning)
    if m:
        result["pmi"] = float(m.group(1))

    # If new format matched, return early
    if result:
        return result

    # ── Old COT format: "COT EUR lev_funds=+39% -> GBPUSD BUY" ──
    m = re.search(r'COT\s+\w+\s+\w+=([+-]?\d+)%', reasoning)
    if m:
        result["base_cot_net_pct"] = float(m.group(1))

    m = re.search(r'price[_ ]?trend[= ]+(-?[0-9.]+)%?', reasoning, re.IGNORECASE)
    if m:
        result["price_trend_pct"] = float(m.group(1))

    # If it's technical format, parse those features
    if "RSI=" in reasoning:
        tech = parse_technical_reasoning(reasoning)
        result.update(tech)

    return result


def parse_news_reasoning(reasoning: str) -> Dict:
    """
    Parse a news agent reasoning string.

    Formats:
      Old rich: "EUR: +37/-49 (28 art) | USD: +68/-75 (39 art) | score=-5"
      New compact: "EUR:+15/-21 USD:+22/-35"
      Early (technical fallback): "RSI=39.79 (oversold), MACD cross=bearish (-0.000249)"
    """
    result = {}

    # ── New compact format (Mar 13+): "EUR:+15/-21 USD:+22/-35" ──
    # No spaces after colon, no article counts, no score
    compact_blocks = re.findall(
        r'(\w+):([+-]?\d+)/([+-]?\d+)', reasoning
    )
    if len(compact_blocks) >= 1 and "(art)" not in reasoning and "art)" not in reasoning:
        _, bull, bear = compact_blocks[0]
        result["base_bull_count"] = abs(int(bull))
        result["base_bear_count"] = abs(int(bear))
        # Estimate article count from bull+bear counts
        result["base_article_count"] = result["base_bull_count"] + result["base_bear_count"]
    if len(compact_blocks) >= 2:
        _, bull, bear = compact_blocks[1]
        result["quote_bull_count"] = abs(int(bull))
        result["quote_bear_count"] = abs(int(bear))
        result["quote_article_count"] = result["quote_bull_count"] + result["quote_bear_count"]

    if result:
        # Compute sentiment score from bull/bear diff
        base_net = result.get("base_bull_count", 0) - result.get("base_bear_count", 0)
        quote_net = result.get("quote_bull_count", 0) - result.get("quote_bear_count", 0)
        result["sentiment_score"] = base_net - quote_net
        return result

    # ── Old rich format: "EUR: +37/-49 (28 art) | USD: +68/-75 (39 art) | score=-5" ──
    currency_blocks = re.findall(
        r'(\w+):\s*\+(\d+)/\-(\d+)\s*\((\d+)\s*art\)', reasoning
    )
    if len(currency_blocks) >= 1:
        _, bull, bear, total = currency_blocks[0]
        result["base_bull_count"] = int(bull)
        result["base_bear_count"] = int(bear)
        result["base_article_count"] = int(total)
    if len(currency_blocks) >= 2:
        _, bull, bear, total = currency_blocks[1]
        result["quote_bull_count"] = int(bull)
        result["quote_bear_count"] = int(bear)
        result["quote_article_count"] = int(total)

    # Score
    m = re.search(r'score=([+-]?\d+)', reasoning)
    if m:
        result["sentiment_score"] = int(m.group(1))

    # Simple bull/bear counts (fallback)
    if "base_bull_count" not in result:
        m = re.search(r'bull[= ]+(\d+)', reasoning, re.IGNORECASE)
        if m:
            result["base_bull_count"] = int(m.group(1))
        m = re.search(r'bear[= ]+(\d+)', reasoning, re.IGNORECASE)
        if m:
            result["base_bear_count"] = int(m.group(1))

    # If technical format, parse those
    if "RSI=" in reasoning:
        tech = parse_technical_reasoning(reasoning)
        result.update(tech)

    return result


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL GROUPING — aggregate signals into pseudo-snapshots
# ═══════════════════════════════════════════════════════════════════════

def load_signals(signals_file: str, assets: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """
    Load signals from JSONL file, grouped by asset.

    Returns: {asset: [signal_dicts sorted by timestamp]}
    """
    signals_by_asset = defaultdict(list)
    bad_lines = 0

    with open(signals_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sig = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            asset = sig.get("asset", "")
            if assets and asset not in assets:
                continue

            signals_by_asset[asset].append(sig)

    # Sort each asset's signals by timestamp
    for asset in signals_by_asset:
        signals_by_asset[asset].sort(key=lambda s: s.get("timestamp", ""))

    if bad_lines:
        print(f"  (skipped {bad_lines} malformed JSON lines)")

    return dict(signals_by_asset)


def group_into_snapshots(signals: List[Dict],
                         window_seconds: int = 600) -> List[Dict]:
    """
    Group signals into pseudo-snapshots by clustering within time windows.

    Signals within `window_seconds` of each other are treated as a single
    snapshot (they came from the same polling cycle). Each group must have
    at least 2 agent types to be useful.

    Returns list of pseudo-snapshot dicts with the same structure as
    the live Redis snapshots.
    """
    if not signals:
        return []

    snapshots = []
    current_group = [signals[0]]
    group_time = parse_timestamp(signals[0]["timestamp"])

    for sig in signals[1:]:
        sig_time = parse_timestamp(sig["timestamp"])
        gap = (sig_time - group_time).total_seconds()

        if gap <= window_seconds:
            current_group.append(sig)
        else:
            # Flush current group
            snap = _build_pseudo_snapshot(current_group)
            if snap:
                snapshots.append(snap)
            current_group = [sig]
            group_time = sig_time

    # Flush last group
    snap = _build_pseudo_snapshot(current_group)
    if snap:
        snapshots.append(snap)

    return snapshots


def _build_pseudo_snapshot(signal_group: List[Dict]) -> Optional[Dict]:
    """
    Convert a group of signals (same time window) into a pseudo-snapshot
    compatible with the existing pipeline.

    The snapshot structure mirrors what the live Redis capturer produces:
    {
      "captured_at": ISO timestamp,
      "asset": str,
      "agents_active": int,
      "agents": {
        "technical": {...},
        "macro": {...},
        "news": {...},
        "fundamental": {...},
      }
    }
    """
    if not signal_group:
        return None

    asset = signal_group[0]["asset"]
    timestamp = signal_group[0]["timestamp"]

    # Parse each signal's reasoning and assign to agent type
    agents = {}

    for sig in signal_group:
        agent_id = sig.get("agent_id", "")
        reasoning = sig.get("reasoning", "")
        direction = sig.get("direction", "HOLD")
        confidence = sig.get("confidence", 0.5)

        if "technical" in agent_id:
            parsed = parse_technical_reasoning(reasoning)
            parsed["agent_id"] = agent_id
            parsed["signal_type"] = "technical"
            parsed["direction"] = direction
            parsed["confidence"] = confidence
            # Merge (don't overwrite if we already have one, prefer higher confidence)
            if "technical" not in agents or confidence > agents["technical"].get("confidence", 0):
                agents["technical"] = parsed

        elif "macro" in agent_id:
            parsed = parse_macro_reasoning(reasoning)
            parsed["agent_id"] = agent_id
            parsed["signal_type"] = "macro"
            parsed["direction"] = direction
            parsed["confidence"] = confidence
            # If macro had technical format, store under technical if needed
            if "rsi" in parsed and "technical" not in agents:
                agents["technical"] = parsed.copy()
            agents["macro"] = parsed

        elif "fundamental" in agent_id:
            parsed = parse_fundamental_reasoning(reasoning)
            parsed["agent_id"] = agent_id
            parsed["signal_type"] = "fundamental"
            parsed["direction"] = direction
            parsed["confidence"] = confidence
            if "rsi" in parsed and "technical" not in agents:
                agents["technical"] = parsed.copy()
            agents["fundamental"] = parsed

        elif "news" in agent_id:
            parsed = parse_news_reasoning(reasoning)
            parsed["agent_id"] = agent_id
            parsed["signal_type"] = "news"
            parsed["direction"] = direction
            parsed["confidence"] = confidence
            if "rsi" in parsed and "technical" not in agents:
                agents["technical"] = parsed.copy()
            agents["news"] = parsed

    # Need at least a technical signal for momentum features
    if "technical" not in agents:
        return None

    return {
        "captured_at": timestamp,
        "asset": asset,
        "agents_active": len(agents),
        "agents": agents,
        "source": "postgres_export",
    }


# ═══════════════════════════════════════════════════════════════════════
# DEDUPLICATION & SPLITTING (same as snapshot_to_dataset.py)
# ═══════════════════════════════════════════════════════════════════════

def deduplicate_snapshots(snapshots: List[Dict],
                          min_interval_seconds: int = 300) -> List[Dict]:
    """Remove near-duplicate snapshots within min_interval_seconds."""
    if not snapshots:
        return []
    deduped = [snapshots[0]]
    last_time = parse_timestamp(snapshots[0]["captured_at"])
    for snap in snapshots[1:]:
        snap_time = parse_timestamp(snap["captured_at"])
        gap = (snap_time - last_time).total_seconds()
        if gap >= min_interval_seconds:
            deduped.append(snap)
            last_time = snap_time
    return deduped


def split_by_time_blocks(dataset: List[Dict],
                         train_ratio: float = 0.75,
                         block_hours: int = 4,
                         gap_hours: int = 4) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train/val using non-overlapping temporal blocks.
    Enforces a time gap between train and val to eliminate label leakage.
    """
    if not dataset:
        return [], []

    sorted_data = sorted(dataset, key=lambda d: d["timestamp"])
    first_ts = parse_timestamp(sorted_data[0]["timestamp"])

    # Assign each sample to a block
    blocks = defaultdict(list)
    for sample in sorted_data:
        ts = parse_timestamp(sample["timestamp"])
        hours_since_start = (ts - first_ts).total_seconds() / 3600
        block_id = int(hours_since_start // block_hours)
        blocks[block_id].append(sample)

    block_ids = sorted(blocks.keys())
    n_blocks = len(block_ids)

    if n_blocks < 3:
        split_idx = max(1, int(len(sorted_data) * train_ratio))
        return sorted_data[:split_idx], sorted_data[split_idx:]

    # Split blocks
    n_train_blocks = max(1, int(n_blocks * train_ratio))
    train_block_ids = block_ids[:n_train_blocks]
    last_train_block = train_block_ids[-1]

    # Gap: skip blocks within gap_hours of the last train block
    gap_blocks = max(1, int(gap_hours / block_hours))
    first_val_block = last_train_block + 1 + gap_blocks
    val_block_ids = [b for b in block_ids if b >= first_val_block]

    train_data = []
    for bid in train_block_ids:
        train_data.extend(blocks[bid])

    val_data = []
    for bid in val_block_ids:
        val_data.extend(blocks[bid])

    return train_data, val_data


# ═══════════════════════════════════════════════════════════════════════
# MAIN CONVERSION
# ═══════════════════════════════════════════════════════════════════════

def convert_asset(asset: str, price_bars: List[Dict],
                  snapshots: List[Dict], forward_hours: int = 4,
                  dedup_interval: int = 300,
                  min_move_pips: int = 0) -> Tuple[List[Dict], Dict]:
    """Convert pseudo-snapshots for one asset into labeled training rows."""
    builder = QuantumVectorBuilder(window_size=252)

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
        captured_at = parse_timestamp(snap["captured_at"])
        agents = snap.get("agents", {})

        technical = agents.get("technical") or {}
        macro = agents.get("macro") or {}
        news = agents.get("news") or {}
        fundamental = agents.get("fundamental") or {}
        risk_sentiment = agents.get("risk_sentiment") or {}
        commodity = agents.get("commodity") or {}

        # If technical doesn't have price, try to get it from price_bars
        if "price" not in technical and price_bars:
            p = find_price_at(price_bars, captured_at, max_gap_hours=2.0)
            if p is not None:
                technical["price"] = p

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

        # Fix 3: Noise filter
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
            "source": "postgres_export",
        })

        stats["labeled"] += 1

    if stats["labeled"] > 0:
        positives = sum(1 for d in dataset if d["label"] == 1)
        stats["label_balance"] = positives / stats["labeled"]

    return dataset, stats


# ═══════════════════════════════════════════════════════════════════════
# INSPECTION
# ═══════════════════════════════════════════════════════════════════════

def inspect_dataset(dataset: List[Dict], metadata: Dict):
    """Print detailed statistics about the dataset."""
    if not dataset:
        print("\n  *** EMPTY DATASET ***\n")
        return

    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples:  {len(dataset)}")
    print(f"  Train samples:  {metadata.get('train_count', '?')}")
    print(f"  Val samples:    {metadata.get('val_count', '?')}")
    print(f"  Assets:         {metadata.get('assets', '?')}")
    print(f"  Source:         PostgreSQL signal export")

    # Label balance
    labels = [d["label"] for d in dataset]
    pos = sum(labels)
    print(f"  Label balance:  {pos}/{len(labels)} = {pos/len(labels):.1%} positive")

    # Feature stats for 8Q
    FEATURE_NAMES_8Q = [
        "momentum_composite", "realized_vol_20d", "momentum_acceleration",
        "atr_normalized", "event_risk_score", "net_sentiment",
        "sentiment_momentum", "risk_sentiment",
    ]

    print(f"\n  8Q Feature Statistics:")
    print(f"  {'Feature':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Unique':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

    for i, name in enumerate(FEATURE_NAMES_8Q):
        vals = [d["features_8q"][i] for d in dataset]
        arr = np.array(vals)
        unique = len(set(round(v, 6) for v in vals))
        print(f"  {name:<25} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{arr.min():>8.4f} {arr.max():>8.4f} {unique:>7}")

    # 14Q slot check for fast features
    SLOT_NAMES = {6: "price_momentum_4h (slot 6)", 12: "intraday_vol (slot 12)"}
    print(f"\n  Fast Feature Slots (14Q):")
    for slot, name in SLOT_NAMES.items():
        vals = [d["features_14q"][slot] for d in dataset]
        arr = np.array(vals)
        unique = len(set(round(v, 6) for v in vals))
        zeros = sum(1 for v in vals if abs(v) < 1e-8)
        print(f"  {name:<30} mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"unique={unique}  zeros={zeros}/{len(vals)}")

    # Time range
    timestamps = sorted([d["timestamp"] for d in dataset])
    print(f"\n  Time range: {timestamps[0][:16]} → {timestamps[-1][:16]}")

    # Per-asset breakdown
    asset_counts = defaultdict(int)
    for d in dataset:
        asset_counts[d["asset"]] += 1
    print(f"\n  Per-asset counts:")
    for asset, count in sorted(asset_counts.items(), key=lambda x: -x[1]):
        print(f"    {asset}: {count}")

    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert PostgreSQL signal exports to VQC training dataset"
    )
    parser.add_argument("--signals-file", type=str, default=str(DEFAULT_SIGNALS_FILE),
                        help="Path to signals_export.jsonl")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Assets to convert (default: all in file)")
    parser.add_argument("--forward-hours", type=int, default=4,
                        help="Forward horizon for label (default: 4h)")
    parser.add_argument("--dedup-interval", type=int, default=300,
                        help="Min seconds between snapshots (default: 300)")
    parser.add_argument("--group-window", type=int, default=600,
                        help="Max seconds to group signals into one snapshot (default: 60)")
    parser.add_argument("--min-move-pips", type=int, default=10,
                        help="Noise filter: min forward move in pips (default: 10, 0=disabled)")
    parser.add_argument("--block-hours", type=int, default=4,
                        help="Block size for temporal split (default: 4h)")
    parser.add_argument("--gap-hours", type=int, default=4,
                        help="Gap between train/val blocks (default: 4h)")
    parser.add_argument("--train-ratio", type=float, default=0.75,
                        help="Train/val split ratio (default: 0.75)")
    parser.add_argument("--inspect", action="store_true",
                        help="Print stats without saving")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: results/historical_dataset.json)")
    parser.add_argument("--no-block-split", action="store_true",
                        help="Use simple time split instead of block split")
    parser.add_argument("--no-yahoo", action="store_true",
                        help="Skip Yahoo fetch (use for testing where Yahoo is blocked)")

    args = parser.parse_args()

    signals_file = args.signals_file
    if not os.path.exists(signals_file):
        print(f"ERROR: Signals file not found: {signals_file}")
        print(f"  Expected at: {DEFAULT_SIGNALS_FILE}")
        sys.exit(1)

    print(f"Loading signals from: {signals_file}")
    signals_by_asset = load_signals(signals_file, args.assets)

    if not signals_by_asset:
        print("ERROR: No signals loaded")
        sys.exit(1)

    assets = sorted(signals_by_asset.keys())
    print(f"Found {sum(len(v) for v in signals_by_asset.values()):,} signals "
          f"across {len(assets)} assets: {', '.join(assets)}")

    # Group signals into pseudo-snapshots
    print(f"\nGrouping signals into snapshots (window={args.group_window}s)...")
    snapshots_by_asset = {}
    for asset in assets:
        snaps = group_into_snapshots(signals_by_asset[asset],
                                     window_seconds=args.group_window)
        snapshots_by_asset[asset] = snaps
        print(f"  {asset}: {len(signals_by_asset[asset]):,} signals → "
              f"{len(snaps)} pseudo-snapshots")

    # Fetch price data
    print(f"\nFetching price data from Yahoo Finance...")
    price_data = {}
    for asset in assets:
        if args.no_yahoo:
            print(f"  {asset}: SKIPPED (--no-yahoo)")
            price_data[asset] = []
            continue

        # Determine how far back we need
        snaps = snapshots_by_asset[asset]
        if not snaps:
            price_data[asset] = []
            continue

        first_ts = parse_timestamp(snaps[0]["captured_at"])
        last_ts = parse_timestamp(snaps[-1]["captured_at"])
        span_days = (last_ts - first_ts).days + 7  # +7 for forward labels + padding

        # Yahoo max for 1h bars is usually 60d; request what we need
        period = f"{min(span_days + 14, 59)}d"
        print(f"  {asset}: fetching {period} of 1h bars...", end=" ", flush=True)
        bars = fetch_price_series(asset, period=period, interval="1h")
        if bars:
            print(f"OK ({len(bars)} bars, {bars[0]['timestamp'].strftime('%m/%d')} → "
                  f"{bars[-1]['timestamp'].strftime('%m/%d')})")
            price_data[asset] = bars
        else:
            print("FAILED")
            price_data[asset] = []

    # Convert each asset
    print(f"\nConverting to training dataset (forward={args.forward_hours}h, "
          f"noise_filter={args.min_move_pips}pips)...")

    all_dataset = []
    all_stats = []

    for asset in assets:
        if not snapshots_by_asset[asset]:
            print(f"  {asset}: no snapshots, skipping")
            continue
        if not price_data.get(asset):
            print(f"  {asset}: no price data, skipping")
            continue

        dataset, stats = convert_asset(
            asset=asset,
            price_bars=price_data[asset],
            snapshots=snapshots_by_asset[asset],
            forward_hours=args.forward_hours,
            dedup_interval=args.dedup_interval,
            min_move_pips=args.min_move_pips,
        )

        all_dataset.extend(dataset)
        all_stats.append(stats)

        noise_str = f", noise={stats['skipped_noise']}" if args.min_move_pips > 0 else ""
        print(f"  {asset}: {stats['total_snapshots']} snaps → "
              f"{stats['valid_vectors']} vectors → {stats['labeled']} labeled "
              f"(balance={stats['label_balance']:.1%}"
              f", no_price={stats['skipped_no_price']}"
              f", build_err={stats['skipped_build_error']}"
              f"{noise_str})")

    if not all_dataset:
        print("\nERROR: No samples produced. Check price data availability.")
        sys.exit(1)

    # Sort by timestamp for splitting
    all_dataset.sort(key=lambda d: d["timestamp"])

    # Split
    if args.no_block_split:
        split_idx = int(len(all_dataset) * args.train_ratio)
        train_data = all_dataset[:split_idx]
        val_data = all_dataset[split_idx:]
        split_method = "simple_time"
    else:
        train_data, val_data = split_by_time_blocks(
            all_dataset,
            train_ratio=args.train_ratio,
            block_hours=args.block_hours,
            gap_hours=args.gap_hours,
        )
        split_method = "temporal_block"

    print(f"\nSplit ({split_method}): {len(train_data)} train / {len(val_data)} val")

    # Metadata
    metadata = {
        "source": "postgres_signal_export",
        "signals_file": os.path.basename(signals_file),
        "total_signals": sum(len(v) for v in signals_by_asset.values()),
        "total_samples": len(all_dataset),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "assets": assets,
        "forward_hours": args.forward_hours,
        "min_move_pips": args.min_move_pips,
        "split_method": split_method,
        "block_hours": args.block_hours,
        "gap_hours": args.gap_hours,
        "dedup_interval": args.dedup_interval,
        "group_window": args.group_window,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "per_asset_stats": all_stats,
    }

    # Inspect
    inspect_dataset(all_dataset, metadata)

    if args.inspect:
        print("  (--inspect mode, not saving)")
        return

    # Save
    output_file = args.output or str(RESULTS_DIR / "historical_dataset.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output = {
        "metadata": metadata,
        "train": train_data,
        "val": val_data,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved: {output_file} ({size_mb:.1f} MB)")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
