#!/usr/bin/env python3
"""
Redis Signal Capture Tool
==========================
Captures live agent payloads from Redis and saves them as structured
snapshots for offline validation through the quantum_signal pipeline.

REQUIRES: pip install redis
(This is the ONLY external dependency in the entire phase0 toolchain.)

Two capture modes:
  1. POLL mode (default): reads cached signal keys every N seconds.
     More reliable — catches signals even if capture starts after publish.
  2. PUBSUB mode: subscribes to signal channels for real-time capture.
     Lower latency but misses signals published before subscription.

Storage format:
  snapshots/
    EURUSD/
      2026-04-04T09-15-00Z.json    # one file per capture window
      2026-04-04T09-30-00Z.json
    GBPUSD/
      ...
    _raw/                           # raw per-agent dumps (for debugging)
      technical_EURUSD_2026-04-04T09-15-00Z.json

Each snapshot JSON contains:
  {
    "captured_at": "ISO timestamp",
    "asset": "EURUSD",
    "capture_mode": "poll",
    "agents": {
      "technical": { ... raw payload ... },
      "macro": { ... },
      "news": { ... },
      "fundamental": { ... },
      "risk_sentiment": { ... },
      "commodity": { ... }
    },
    "staleness": {
      "technical": 12.5,  // seconds since agent published
      "macro": 45.2,
      ...
    }
  }

Usage:
  # Poll mode (recommended): capture every 60 seconds for all assets
  python scripts/redis_capture.py --mode poll --interval 60 --duration 3600

  # Poll once and exit (snapshot mode)
  python scripts/redis_capture.py --mode poll --once

  # Pub/sub mode: listen for live signals
  python scripts/redis_capture.py --mode pubsub --duration 3600

  # Custom Redis host (e.g., Docker on port 6380)
  python scripts/redis_capture.py --redis-host localhost --redis-port 6380

  # Specific assets only
  python scripts/redis_capture.py --assets EURUSD,GBPUSD
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default assets (matches the real system's WATCHED_PAIRS + equities)
DEFAULT_ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "SPY", "NVDA", "AAPL", "MSFT", "TSLA"]

# Redis key patterns (from technical_agent.py, macro_agent.py, etc.)
SIGNAL_KEYS = {
    "technical": "signal:technical:{asset}",
    "macro": "signal:macro:{asset}",
    "news": "signal:news:{asset}",
    "fundamental": "signal:fundamental:{asset}",
}
# Global keys (not per-asset)
GLOBAL_KEYS = {
    "risk_sentiment": "signal:risk_sentiment",
    "commodity": "signal:commodity",
}

# Pub/sub channels
SIGNAL_CHANNELS = [
    "signals:technical",
    "signals:macro",
    "signals:news",
    "signals:fundamental",
    "signals:risk_sentiment",
    "signals:commodity",
]


def connect_redis(host="localhost", port=6379, db=0):
    """Connect to Redis. Requires redis-py package."""
    try:
        import redis
    except ImportError:
        print("ERROR: redis-py not installed. Run: pip install redis")
        print("This is the only external dependency needed for capture.")
        sys.exit(1)

    r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    try:
        r.ping()
        logger.info(f"Connected to Redis at {host}:{port}")
        return r
    except Exception as e:
        logger.error(f"Cannot connect to Redis at {host}:{port}: {e}")
        sys.exit(1)


def poll_snapshot(r, assets, now=None):
    """
    Read all cached signal keys from Redis for each asset.
    Returns a dict: {asset: snapshot_dict}
    """
    if now is None:
        now = datetime.now(timezone.utc)

    snapshots = {}

    for asset in assets:
        agents = {}
        staleness = {}

        # Per-asset signal keys
        for agent_name, key_pattern in SIGNAL_KEYS.items():
            key = key_pattern.format(asset=asset)
            try:
                raw = r.get(key)
                if raw:
                    data = json.loads(raw)
                    agents[agent_name] = data
                    # Calculate staleness
                    ts_str = data.get("timestamp", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            staleness[agent_name] = round((now - ts).total_seconds(), 1)
                        except Exception:
                            staleness[agent_name] = -1  # unparseable
                else:
                    agents[agent_name] = None
                    staleness[agent_name] = None
            except Exception as e:
                logger.warning(f"Error reading {key}: {e}")
                agents[agent_name] = None
                staleness[agent_name] = None

        # Global keys (same for all assets)
        for agent_name, key in GLOBAL_KEYS.items():
            try:
                raw = r.get(key)
                if raw:
                    data = json.loads(raw)
                    agents[agent_name] = data
                    ts_str = data.get("timestamp", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            staleness[agent_name] = round((now - ts).total_seconds(), 1)
                        except Exception:
                            staleness[agent_name] = -1
                else:
                    agents[agent_name] = None
                    staleness[agent_name] = None
            except Exception as e:
                logger.warning(f"Error reading {key}: {e}")

        # Count how many agents have data
        active = sum(1 for v in agents.values() if v is not None)

        if active > 0:
            snapshots[asset] = {
                "captured_at": now.isoformat(),
                "asset": asset,
                "capture_mode": "poll",
                "agents_active": active,
                "agents_total": len(agents),
                "agents": agents,
                "staleness": staleness,
            }
            logger.info(f"  {asset}: {active}/{len(agents)} agents active")
        else:
            logger.info(f"  {asset}: no data in Redis")

    return snapshots


def save_snapshots(snapshots, output_dir):
    """Save snapshots to disk in the structured format."""
    output_path = Path(output_dir)
    saved = 0

    for asset, snapshot in snapshots.items():
        # Per-asset directory
        asset_dir = output_path / asset
        asset_dir.mkdir(parents=True, exist_ok=True)

        # Filename from timestamp
        ts = snapshot["captured_at"].replace(":", "-").replace("+", "_")
        filename = f"{ts}.json"

        filepath = asset_dir / filename
        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2)
        saved += 1

        # Also save to _raw for individual agent inspection
        raw_dir = output_path / "_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for agent_name, payload in snapshot["agents"].items():
            if payload:
                raw_file = raw_dir / f"{agent_name}_{asset}_{ts}.json"
                with open(raw_file, "w") as f:
                    json.dump(payload, f, indent=2)

    return saved


def run_poll_mode(r, assets, output_dir, interval=60, duration=None, once=False):
    """
    Poll Redis every `interval` seconds, save snapshots.
    If `once` is True, poll once and exit.
    If `duration` is set, run for that many seconds.
    """
    start_time = time.time()
    total_saved = 0
    tick = 0

    while True:
        tick += 1
        logger.info(f"─── Poll tick #{tick} ───")
        now = datetime.now(timezone.utc)

        snapshots = poll_snapshot(r, assets, now)
        if snapshots:
            saved = save_snapshots(snapshots, output_dir)
            total_saved += saved
            logger.info(f"Saved {saved} snapshots (total: {total_saved})")
        else:
            logger.info("No agent data found in Redis.")

        if once:
            break

        if duration and (time.time() - start_time) >= duration:
            logger.info(f"Duration limit reached ({duration}s). Stopping.")
            break

        logger.info(f"Waiting {interval}s...")
        time.sleep(interval)

    return total_saved


def run_pubsub_mode(r, assets, output_dir, duration=None):
    """
    Subscribe to signal channels and save payloads as they arrive.
    Groups by asset and saves periodic snapshots.
    """
    import redis as redis_module  # need PubSub type

    pubsub = r.pubsub()
    pubsub.subscribe(*SIGNAL_CHANNELS)
    logger.info(f"Subscribed to: {', '.join(SIGNAL_CHANNELS)}")

    start_time = time.time()
    # Buffer: {asset: {agent_name: latest_payload}}
    buffer = {}
    last_flush = time.time()
    flush_interval = 30  # seconds
    total_saved = 0

    try:
        for message in pubsub.listen():
            if message["type"] != "message":
                continue

            channel = message["channel"]
            try:
                data = json.loads(message["data"])
            except json.JSONDecodeError:
                continue

            asset = data.get("asset", "UNKNOWN")
            # Determine agent type from channel
            agent_type = channel.split(":")[-1] if ":" in channel else channel

            if asset not in buffer:
                buffer[asset] = {"agents": {}, "first_seen": time.time()}
            buffer[asset]["agents"][agent_type] = data
            logger.debug(f"Received {agent_type} for {asset}")

            # Periodic flush
            if time.time() - last_flush >= flush_interval:
                now = datetime.now(timezone.utc)
                snapshots = {}
                for a, buf in buffer.items():
                    snapshots[a] = {
                        "captured_at": now.isoformat(),
                        "asset": a,
                        "capture_mode": "pubsub",
                        "agents_active": len(buf["agents"]),
                        "agents_total": len(SIGNAL_KEYS) + len(GLOBAL_KEYS),
                        "agents": buf["agents"],
                        "staleness": {},
                    }
                if snapshots:
                    saved = save_snapshots(snapshots, output_dir)
                    total_saved += saved
                    logger.info(f"Flushed {saved} snapshots (total: {total_saved})")
                buffer.clear()
                last_flush = time.time()

            if duration and (time.time() - start_time) >= duration:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        pubsub.unsubscribe()

    return total_saved


def main():
    parser = argparse.ArgumentParser(
        description="Capture live Redis agent signals for quantum_signal validation"
    )
    parser.add_argument("--mode", choices=["poll", "pubsub"], default="poll",
                        help="Capture mode (default: poll)")
    parser.add_argument("--redis-host", default="localhost",
                        help="Redis host (default: localhost)")
    parser.add_argument("--redis-port", type=int, default=6379,
                        help="Redis port (default: 6379)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: snapshots/)")
    parser.add_argument("--assets", default=None,
                        help="Comma-separated asset list (default: all)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Poll interval in seconds (default: 60)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Total capture duration in seconds (default: unlimited)")
    parser.add_argument("--once", action="store_true",
                        help="Poll once and exit")
    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        phase0_dir = Path(__file__).parent.parent
        args.output_dir = str(phase0_dir / "snapshots")

    assets = args.assets.split(",") if args.assets else DEFAULT_ASSETS

    r = connect_redis(args.redis_host, args.redis_port)

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Assets: {', '.join(assets)}")
    logger.info(f"Output: {args.output_dir}")

    if args.mode == "poll":
        total = run_poll_mode(r, assets, args.output_dir,
                              interval=args.interval,
                              duration=args.duration,
                              once=args.once)
    else:
        total = run_pubsub_mode(r, assets, args.output_dir,
                                duration=args.duration)

    logger.info(f"Capture complete. Total snapshots saved: {total}")
    logger.info(f"Snapshots in: {args.output_dir}")


if __name__ == "__main__":
    main()
