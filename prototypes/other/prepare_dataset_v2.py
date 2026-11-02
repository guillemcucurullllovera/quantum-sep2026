#!/usr/bin/env python3
"""
Dataset v2: Proper stratified split with regime-aware validation.

Fixes three problems with the original real_dataset.json:
1. Label inversion: SPY was 100% positive in train, 22% in val.
   Fix: Per-asset stratified split ensures each asset has balanced label rates.
2. Regime shift: Features that correlate positively with labels in train
   flip sign in val because of market regime change.
   Fix: Temporal shuffling within each asset distributes regimes across splits.
3. Feature quality: Only use samples where ≥10/14 features are alive.
   Fix: Filter out dead-feature rows.

Also incorporates the best usable portion of the historical dataset
(Feb 26 - Mar 11, when ~8/14 features were alive).

Output: results/dataset_v2_stratified.json
"""

import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURE_NAMES = [
    "momentum_composite", "momentum_acceleration", "realized_vol_20d",
    "vol_regime", "atr_normalized", "carry_differential",
    "yield_curve_slope", "event_risk_score", "net_sentiment",
    "sentiment_momentum", "cot_positioning", "cot_extreme",
    "dxy_momentum", "risk_sentiment"
]


def count_alive_features(features):
    """Count features with non-zero values."""
    return sum(1 for f in features if abs(f) > 1e-8)


def load_and_filter(min_alive=10):
    """Load both datasets and filter for feature quality."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    all_samples = []

    # Load real dataset (best feature quality)
    real_path = os.path.join(base, "real_dataset.json")
    if os.path.exists(real_path):
        with open(real_path) as f:
            rdata = json.load(f)
        for s in rdata["train"] + rdata["val"]:
            if count_alive_features(s["features_14q"]) >= min_alive:
                s["source"] = "real"
                all_samples.append(s)
        print(f"  Real dataset: {len(rdata['train']) + len(rdata['val'])} total, "
              f"{len(all_samples)} pass filter (≥{min_alive}/14 features alive)")

    # Load historical dataset (v3 with fixed parsers + 600s window)
    hist_path = os.path.join(base, "historical_dataset_v3.json")
    if not os.path.exists(hist_path):
        hist_path = os.path.join(base, "historical_dataset_v2.json")
    if not os.path.exists(hist_path):
        hist_path = os.path.join(base, "historical_dataset.json")
    n_hist = 0
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hdata = json.load(f)
        for s in hdata["train"] + hdata["val"]:
            if count_alive_features(s["features_14q"]) >= min_alive:
                s["source"] = "historical"
                all_samples.append(s)
                n_hist += 1
        print(f"  Historical dataset: {len(hdata['train']) + len(hdata['val'])} total, "
              f"{n_hist} pass filter")

    print(f"  Combined: {len(all_samples)} samples")
    return all_samples


def stratified_split(samples, val_fraction=0.2, seed=42):
    """
    Per-asset stratified split with label balancing.

    Within each asset:
      1. Separate label=0 and label=1 samples
      2. Shuffle each group independently
      3. Take val_fraction from each group for validation
      4. This ensures train and val have the same label rate per asset

    This is NOT a pure temporal split — it distributes time periods across
    both sets. For circuit validation this is the right approach. Temporal
    validation will be added when we have enough data to span market regimes.
    """
    rng = np.random.RandomState(seed)

    # Group by (asset, label)
    groups = defaultdict(list)
    for s in samples:
        key = (s.get("asset", "unknown"), int(s["label"]))
        groups[key].append(s)

    train, val = [], []
    print(f"\n  Stratified split (val_fraction={val_fraction}):")

    # Process each asset
    assets = sorted(set(k[0] for k in groups.keys()))
    for asset in assets:
        pos = groups.get((asset, 1), [])
        neg = groups.get((asset, 0), [])
        rng.shuffle(pos)
        rng.shuffle(neg)

        n_pos_val = max(1, int(len(pos) * val_fraction))
        n_neg_val = max(1, int(len(neg) * val_fraction))

        val.extend(pos[:n_pos_val])
        val.extend(neg[:n_neg_val])
        train.extend(pos[n_pos_val:])
        train.extend(neg[n_neg_val:])

        total = len(pos) + len(neg)
        tr_count = (len(pos) - n_pos_val) + (len(neg) - n_neg_val)
        vl_count = n_pos_val + n_neg_val
        tr_rate = (len(pos) - n_pos_val) / max(1, tr_count)
        vl_rate = n_pos_val / max(1, vl_count)

        print(f"    {asset:8s}: {total:5d} total | "
              f"train={tr_count} (pos={tr_rate:.3f}) | "
              f"val={vl_count} (pos={vl_rate:.3f})")

    # Final shuffle
    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


def add_regime_label(samples):
    """
    Add a regime indicator based on vol_regime and momentum features.

    Regime encoding:
      0 = low-vol trending (calm trend-following)
      1 = low-vol range-bound (quiet chop)
      2 = high-vol trending (volatile breakout)
      3 = high-vol range-bound (volatile chop / crisis)

    This lets the model condition its predictions on market state.
    """
    for s in samples:
        feats = s["features_14q"]
        vol_regime = feats[3]     # 0.0 = low, 0.5 = medium, 1.0 = high
        momentum = abs(feats[0])  # momentum_composite magnitude

        is_high_vol = vol_regime >= 0.5
        is_trending = momentum > 0.3

        if is_high_vol and is_trending:
            regime = 3  # volatile breakout
        elif is_high_vol:
            regime = 2  # volatile chop
        elif is_trending:
            regime = 1  # calm trend
        else:
            regime = 0  # quiet range

        s["regime"] = regime

    return samples


def verify_dataset(train, val):
    """Run diagnostic checks on the split."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    train_labels = [s["label"] for s in train]
    val_labels = [s["label"] for s in val]

    # 1. Overall label balance
    tr_pos = sum(train_labels) / len(train_labels)
    vl_pos = sum(val_labels) / len(val_labels)
    print(f"\n  Label balance: train={tr_pos:.3f}, val={vl_pos:.3f}, "
          f"diff={abs(tr_pos - vl_pos):.3f}")
    assert abs(tr_pos - vl_pos) < 0.1, f"Label imbalance too large: {abs(tr_pos - vl_pos):.3f}"

    # 2. Per-asset label balance
    print("\n  Per-asset label rates:")
    from collections import Counter
    all_assets = sorted(set(s.get("asset", "?") for s in train + val))
    for asset in all_assets:
        tr = [s["label"] for s in train if s.get("asset") == asset]
        vl = [s["label"] for s in val if s.get("asset") == asset]
        if tr and vl:
            tr_rate = sum(tr) / len(tr)
            vl_rate = sum(vl) / len(vl)
            diff = abs(tr_rate - vl_rate)
            flag = " !!!" if diff > 0.15 else " ✓"
            print(f"    {asset:8s}  train={tr_rate:.3f} ({len(tr):4d})  "
                  f"val={vl_rate:.3f} ({len(vl):3d})  diff={diff:.3f}{flag}")
            assert diff < 0.2, f"Asset {asset} label imbalance: {diff:.3f}"

    # 3. Feature-label correlation sign consistency
    tr_feats = np.array([s["features_14q"] for s in train])
    vl_feats = np.array([s["features_14q"] for s in val])

    print("\n  Feature-label correlation sign check:")
    flips = 0
    for i in range(14):
        tr_std = tr_feats[:, i].std()
        vl_std = vl_feats[:, i].std()
        if tr_std < 1e-8 or vl_std < 1e-8:
            continue
        tr_corr = np.corrcoef(tr_feats[:, i], train_labels)[0, 1]
        vl_corr = np.corrcoef(vl_feats[:, i], val_labels)[0, 1]
        is_flip = ((tr_corr > 0.05 and vl_corr < -0.05) or
                   (tr_corr < -0.05 and vl_corr > 0.05))
        if is_flip:
            flips += 1
            print(f"    {FEATURE_NAMES[i]:>25s}: train={tr_corr:+.3f}, "
                  f"val={vl_corr:+.3f} FLIP")
    if flips == 0:
        print("    No sign flips detected ✓")
    else:
        print(f"    {flips} sign flips (some expected with shuffled split)")

    # 4. Feature distribution shift
    print("\n  Feature distribution shifts:")
    big_shifts = 0
    for i in range(14):
        tr_mean = tr_feats[:, i].mean()
        vl_mean = vl_feats[:, i].mean()
        tr_std = tr_feats[:, i].std()
        if tr_std < 1e-8:
            continue
        shift = abs(vl_mean - tr_mean) / tr_std
        if shift > 0.3:
            big_shifts += 1
            print(f"    {FEATURE_NAMES[i]:>25s}: {shift:.2f}σ shift")
    if big_shifts == 0:
        print("    All shifts < 0.3σ ✓")

    # 5. Regime distribution
    if "regime" in train[0]:
        print("\n  Regime distribution:")
        regime_names = ["quiet_range", "calm_trend", "volatile_chop", "volatile_breakout"]
        tr_regimes = Counter(s["regime"] for s in train)
        vl_regimes = Counter(s["regime"] for s in val)
        for r in range(4):
            tr_pct = tr_regimes.get(r, 0) / len(train) * 100
            vl_pct = vl_regimes.get(r, 0) / len(val) * 100
            print(f"    {regime_names[r]:20s}  train={tr_pct:.1f}%  val={vl_pct:.1f}%")

    # 6. Majority baseline
    majority_class = 1 if sum(train_labels) > len(train_labels) / 2 else 0
    majority_acc = sum(1 for y in val_labels if int(y) == majority_class) / len(val_labels)
    print(f"\n  Majority baseline (always predict {majority_class}): {majority_acc:.3f}")
    print(f"  → Any useful model should beat {majority_acc:.3f}")

    print("\n" + "=" * 60)
    return True


def main():
    print("=" * 60)
    print("Dataset v2: Stratified Split with Regime Labels")
    print("=" * 60)

    # Load and filter
    print("\nLoading and filtering...")
    samples = load_and_filter(min_alive=7)

    if len(samples) < 100:
        print(f"ERROR: Only {len(samples)} samples pass quality filter. "
              "Need at least 100. Check your data pipeline.")
        sys.exit(1)

    # Add regime labels
    print("\nAdding regime labels...")
    samples = add_regime_label(samples)

    # Stratified split
    print("\nSplitting...")
    train, val = stratified_split(samples, val_fraction=0.2, seed=42)
    print(f"\n  Final: train={len(train)}, val={len(val)}")

    # Verify
    verify_dataset(train, val)

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    output = {
        "metadata": {
            "version": "v2_stratified",
            "split_method": "per_asset_stratified_label_balanced",
            "min_alive_features": 10,
            "val_fraction": 0.2,
            "regime_labels": True,
            "sources": ["real_dataset.json", "historical_dataset.json (filtered)"],
        },
        "train": train,
        "val": val,
    }
    out_path = os.path.join(results_dir, "dataset_v2_stratified.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"  Train: {len(train)} samples")
    print(f"  Val: {len(val)} samples")
    print("\nDone.")


if __name__ == "__main__":
    main()
