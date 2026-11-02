#!/usr/bin/env python3 -u
"""
[HISTORICAL PROTOTYPE — Gen1, preserved verbatim]
Only the two `sys.path`-based imports below were repointed to this prototype's
local copies of its dependencies (`_original_kernel_phase0.py`, `phase1b_train.py`,
both copied verbatim from the same Gen1 source tree). No other logic changed.
NOT independently runnable here: it expects `results/real_dataset.json`
(real agent trading data) which does not exist in this repo and was never
copied in (raw data dumps are excluded per policy). It imports and parses
cleanly; running `main()` requires supplying that dataset file yourself.
The production, currently-maintained PQK implementation is
`algorithms/pqk/` — this file is preserved as development history only.

Phase 2B — Quantum Kernel Training on Real Agent Data
=======================================================
Uses the SAME dataset as phase2a to train quantum kernel SVMs.
No gradient training — just encode, compute Gram matrix, fit SVM.

Tests:
  - Fidelity Quantum Kernel (FQK) with 3 feature maps
  - Projected Quantum Kernel (PQK) with 3 feature maps
  - Multiple SVM regularization values (C)
  - Multiple RBF gamma values (for PQK)

Compares against same baselines as phase2a for direct comparison.

FQK is O(N²) in quantum circuits (slow), PQK is O(N) (fast).
On 1,830 train samples:
  - FQK: ~3.3M kernel elements × 3 feature maps = ~10M circuit evals
  - PQK: ~2,440 circuit evals × 3 feature maps = ~7K circuit evals

Usage:
  python scripts/phase2b_kernel_train.py --dataset results/real_dataset.json
  python scripts/phase2b_kernel_train.py --dataset results/real_dataset.json --fmap ZZFeatureMap
  python scripts/phase2b_kernel_train.py --dataset results/real_dataset.json --pqk-only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required")
    sys.exit(1)

from prototypes.pqk._original_kernel_phase0 import (
    FEATURE_MAPS,
    train_fqk_svm,
    train_pqk_svm,
)

# Import baselines for comparison
from prototypes.pqk.phase1b_train import LogisticBaseline, MajorityBaseline

PHASE0_DIR = Path(__file__).parent.parent
RESULTS_DIR = PHASE0_DIR / "results"


def load_dataset(path: str):
    with open(path) as f:
        data = json.load(f)
    train = data["train"]
    val = data["val"]
    meta = data.get("metadata", {})
    print(f"  Loaded: {len(train)} train / {len(val)} val")
    print(f"  Assets: {meta.get('assets', '?')}")
    return train, val, meta


def run_baselines(train_data, val_data):
    """Quick baselines for comparison table."""
    results = {}

    maj = MajorityBaseline(train_data)
    _, maj_acc = maj.evaluate(val_data)
    results["majority"] = maj_acc

    lr = LogisticBaseline(n_features=8)
    for _ in range(50):
        lr.train_epoch(train_data)
    _, lr_acc = lr.evaluate(val_data)
    results["logistic_regression"] = lr_acc

    print(f"  Majority: {maj_acc:.4f}  |  LR: {lr_acc:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2B: Quantum Kernel Methods")
    parser.add_argument("--dataset", required=True, help="Path to real_dataset.json")
    parser.add_argument("--fmap", nargs="+", default=["ZZFeatureMap", "AngleCNOT", "CPMap"],
                        help="Feature maps to test")
    parser.add_argument("--pqk-only", action="store_true",
                        help="Skip FQK (slow N² computation), only run PQK")
    parser.add_argument("--fqk-only", action="store_true",
                        help="Skip PQK, only run FQK")
    parser.add_argument("--max-fqk-train", type=int, default=500,
                        help="Max train samples for FQK (N² gets slow). Default: 500")
    parser.add_argument("--C", nargs="+", type=float, default=[0.1, 1.0, 10.0],
                        help="SVM C values to test")
    parser.add_argument("--gamma", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0],
                        help="RBF gamma values for PQK")
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 2B — Quantum Kernel Methods on Real Agent Data")
    print("=" * 70)

    train_data, val_data, meta = load_dataset(args.dataset)

    # Extract numpy arrays
    X_train = np.array([d["features_8q"] for d in train_data])
    y_train = np.array([d["label"] for d in train_data])
    X_val = np.array([d["features_8q"] for d in val_data])
    y_val = np.array([d["label"] for d in val_data])

    print(f"  Features shape: {X_train.shape} train, {X_val.shape} val")
    print(f"  Label balance: {y_train.mean():.3f} train, {y_val.mean():.3f} val")

    # Baselines
    print(f"\n{'─'*50}")
    print("  BASELINES (for comparison)")
    print(f"{'─'*50}")
    baseline_results = run_baselines(train_data, val_data)

    all_results = []

    # ── PQK (fast: O(N) quantum circuits) ──
    if not args.fqk_only:
        print(f"\n{'═'*50}")
        print("  PROJECTED QUANTUM KERNEL (PQK)")
        print(f"{'═'*50}")

        for fmap_name in args.fmap:
            if fmap_name not in FEATURE_MAPS:
                print(f"  WARNING: Unknown feature map '{fmap_name}', skipping")
                continue

            for gamma in args.gamma:
                for C in args.C:
                    result = train_pqk_svm(
                        X_train, y_train, X_val, y_val,
                        feature_map_name=fmap_name,
                        gamma=gamma, C=C, verbose=True,
                    )
                    # Remove non-serializable fields
                    result_clean = {k: v for k, v in result.items()
                                    if k not in ("svm", "proj_train", "proj_val")}
                    all_results.append(result_clean)

    # ── FQK (slow: O(N²) quantum circuits) ──
    if not args.pqk_only:
        print(f"\n{'═'*50}")
        print("  FIDELITY QUANTUM KERNEL (FQK)")
        print(f"{'═'*50}")

        # Subsample if train set too large for N²
        if len(X_train) > args.max_fqk_train:
            print(f"\n  NOTE: Subsampling train from {len(X_train)} to "
                  f"{args.max_fqk_train} for FQK (N² kernel computation)")
            idx = np.random.RandomState(42).choice(
                len(X_train), args.max_fqk_train, replace=False
            )
            X_train_fqk = X_train[idx]
            y_train_fqk = y_train[idx]
        else:
            X_train_fqk = X_train
            y_train_fqk = y_train

        for fmap_name in args.fmap:
            if fmap_name not in FEATURE_MAPS:
                continue

            for C in args.C:
                result = train_fqk_svm(
                    X_train_fqk, y_train_fqk, X_val, y_val,
                    feature_map_name=fmap_name,
                    C=C, verbose=True,
                )
                result_clean = {k: v for k, v in result.items() if k != "svm"}
                result_clean["train_samples_used"] = len(X_train_fqk)
                all_results.append(result_clean)

    # ── Results table ──
    print(f"\n{'='*70}")
    print("  PHASE 2B RESULTS — Quantum Kernels on Real Agent Data")
    print(f"{'='*70}")

    # Sort by val accuracy
    all_results.sort(key=lambda r: -r["val_acc"])

    print(f"\n  {'Rank':<5s} {'Type':<6s} {'FeatureMap':<14s} {'C':>6s} "
          f"{'gamma':>6s} {'Train':>8s} {'Val':>8s}")
    print(f"  {'─'*60}")

    for i, r in enumerate(all_results):
        gamma_str = f"{r.get('gamma', '-'):>6.1f}" if 'gamma' in r else f"{'--':>6s}"
        print(f"  {i+1:<5d} {r['kernel_type']:<6s} {r['feature_map']:<14s} "
              f"{r['C']:>6.1f} {gamma_str} {r['train_acc']:>8.4f} {r['val_acc']:>8.4f}")

    # Best results
    if all_results:
        best = all_results[0]
        best_classical = max(baseline_results.values())
        best_classical_name = max(baseline_results, key=baseline_results.get)

        print(f"\n  Best kernel:     {best['kernel_type']} {best['feature_map']} "
              f"C={best['C']} → val={best['val_acc']:.4f}")
        print(f"  Best classical:  {best_classical_name} → val={best_classical:.4f}")
        print(f"  VQC v2 (ref):    val=0.6590 (best)")
        print(f"  Kernel vs LR:    {best['val_acc'] - baseline_results['logistic_regression']:+.4f}")
        print(f"  Kernel vs VQC:   {best['val_acc'] - 0.6590:+.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "2b_quantum_kernels",
        "dataset_meta": meta,
        "baselines": baseline_results,
        "kernel_results": all_results,
        "best_kernel": all_results[0] if all_results else None,
    }

    out_path = RESULTS_DIR / "phase2b_kernel_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
