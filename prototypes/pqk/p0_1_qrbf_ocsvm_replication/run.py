#!/usr/bin/env python3
"""
P0.1 — Replicate Pranjic et al. (arXiv:2411.16970) qrbf-OCSVM on Sparkov.

Paper Table 1, r=0.2, test split, 10-fold CV:
    rbf-OCSVM      F1 = 0.757 ± 0.062
    qrbf-CX        F1 = 0.829 ± 0.043
    qrbf-ECR       F1 = 0.847 ± 0.059
    qrbf-RXX       F1 = 0.827 ± 0.054

Pass criterion: each measured mean within ±5 percentage points of paper.

Usage
-----
    python -m phase0.quantum_qiskit.diagnostics.p0_1_replication.run

The script is idempotent: it caches 1Q-RDM tensors per feature map under
phase0/results/p0_1/ so subsequent runs only re-evaluate the SVM, not the
quantum circuit simulations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ._data import (
    engineer_features,
    load_sparkov_raw,
    subsample_for_anomaly_ratio,
)
from ._kernels import (
    FEATURE_MAPS,
    compute_rdms_for_dataset,
    qrbf_kernel_matrix,
    rbf_kernel_matrix,
    scale_to_pi,
)


# ── paths ───────────────────────────────────────────────────────────

THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[4]
# Cache RDM tensors in a writable scratch dir (workspace mount can't unlink files);
# results JSON still goes to phase0/results/p0_1/ in the workspace.
RESULTS_DIR = WORKSPACE_ROOT / "phase0" / "results" / "p0_1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(os.environ.get("P01_CACHE_DIR", "/tmp/p0_1_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── paper targets ───────────────────────────────────────────────────

PAPER_TABLE_1 = {
    "rbf":      (0.757, 0.062),
    "qrbf-CX":  (0.829, 0.043),
    "qrbf-ECR": (0.847, 0.059),
    "qrbf-RXX": (0.827, 0.054),
}
TOLERANCE = 0.05  # ±5 percentage points

# ── logging ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("p0_1")


# ── metrics ─────────────────────────────────────────────────────────

def f1_score_anomaly(y_true: np.ndarray, y_pred_anomaly: np.ndarray) -> float:
    """F1 with anomalies as the positive class.

    y_true: 1 for anomaly (fraud), 0 for normal.
    y_pred_anomaly: 1 if predicted anomaly, 0 if predicted normal.
    """
    tp = int(((y_pred_anomaly == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_anomaly == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_anomaly == 0) & (y_true == 1)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def ocsvm_predict_anomaly(model: OneClassSVM, X_or_K: np.ndarray) -> np.ndarray:
    """sklearn OneClassSVM returns +1 = inlier, -1 = outlier. Map outlier→1."""
    raw = model.predict(X_or_K)
    return (raw == -1).astype(int)


# ── CV harness ──────────────────────────────────────────────────────

def run_classical_cv(
    X: np.ndarray, y: np.ndarray, gamma: float = 0.1, nu: float = 0.1, k: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """10-fold CV of classical rbf-OCSVM."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train_f1s, test_f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
        # Conventional unsupervised OCSVM: train on the entire training fold
        # (including the few anomalies). This matches paper Section III.A.
        model.fit(Xtr_s)

        train_pred = ocsvm_predict_anomaly(model, Xtr_s)
        test_pred = ocsvm_predict_anomaly(model, Xte_s)

        train_f1s.append(f1_score_anomaly(ytr, train_pred))
        test_f1s.append(f1_score_anomaly(yte, test_pred))

    return {
        "train_mean": float(np.mean(train_f1s)),
        "train_std":  float(np.std(train_f1s)),
        "test_mean":  float(np.mean(test_f1s)),
        "test_std":   float(np.std(test_f1s)),
        "per_fold_test": test_f1s,
    }


def run_quantum_cv(
    X_scaled: np.ndarray, y: np.ndarray, feature_map_name: str,
    gamma: float = 0.1, nu: float = 0.1, k: int = 10, seed: int = 42,
    cache_dir: Path = None,
) -> Dict[str, float]:
    """10-fold CV of qrbf-OCSVM for a specific feature map.

    Pre-computes the 1Q-RDM tensor for all samples ONCE, then builds the
    kernel matrix in each fold from the cached tensor.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_path = cache_dir / f"rdms_{feature_map_name}.npy"
    if cache_path.exists():
        rdms = np.load(cache_path)
        log.info(f"[{feature_map_name}] loaded cached RDMs {rdms.shape} from {cache_path.name}")
    else:
        log.info(f"[{feature_map_name}] computing 1Q-RDMs for {len(X_scaled)} samples...")
        t0 = time.time()
        fm = FEATURE_MAPS[feature_map_name]
        rdms = compute_rdms_for_dataset(X_scaled, fm)
        np.save(cache_path, rdms)
        log.info(f"[{feature_map_name}] RDMs done in {time.time()-t0:.1f}s, cached to {cache_path.name}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train_f1s, test_f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_scaled, y), 1):
        ytr, yte = y[tr_idx], y[te_idx]
        rdms_tr, rdms_te = rdms[tr_idx], rdms[te_idx]

        K_train = qrbf_kernel_matrix(rdms_tr, rdms_tr, gamma=gamma)
        K_test = qrbf_kernel_matrix(rdms_te, rdms_tr, gamma=gamma)

        model = OneClassSVM(kernel="precomputed", nu=nu)
        model.fit(K_train)

        train_pred = ocsvm_predict_anomaly(model, K_train)
        test_pred = ocsvm_predict_anomaly(model, K_test)

        train_f1s.append(f1_score_anomaly(ytr, train_pred))
        test_f1s.append(f1_score_anomaly(yte, test_pred))

    return {
        "train_mean": float(np.mean(train_f1s)),
        "train_std":  float(np.std(train_f1s)),
        "test_mean":  float(np.mean(test_f1s)),
        "test_std":   float(np.std(test_f1s)),
        "per_fold_test": test_f1s,
    }


# ── main ────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--anomaly-ratio", type=float, default=0.2,
                   help="Paper Table 1 main result is at r=0.2")
    p.add_argument("--n-anomalies", type=int, default=100,
                   help="Number of anomaly samples to subsample; 100 → 500 total at r=0.2")
    p.add_argument("--gamma", type=float, default=0.1,
                   help="OCSVM kernel bandwidth (paper: best grid value is 0.1)")
    p.add_argument("--nu", type=float, default=0.1,
                   help="OCSVM nu parameter (paper: best grid value is 0.1)")
    p.add_argument("--cv-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature-maps", nargs="+", default=["CX", "ECR", "RXX"])
    p.add_argument("--skip-quantum", action="store_true",
                   help="Run only the classical baseline (for quick iteration on data)")
    p.add_argument("--invalidate-cache", action="store_true",
                   help="Delete cached 1Q-RDM tensors and recompute")
    args = p.parse_args(argv)

    if args.invalidate_cache:
        for f in CACHE_DIR.glob("rdms_*.npy"):
            try:
                f.unlink()
                log.info(f"deleted cache {f.name}")
            except PermissionError:
                # FUSE mount: zero out instead of unlinking
                f.write_bytes(b"")
                log.info(f"truncated cache {f.name} (FUSE)")

    # ── 1. data acquisition ────────────────────────────────────────
    log.info("loading raw Sparkov dataset...")
    df_raw = load_sparkov_raw(WORKSPACE_ROOT)

    # ── 2. feature engineering ─────────────────────────────────────
    # Use a 75/25 train/test split purely for fraud-rate encoding leakage control,
    # then we re-subsample for the actual OCSVM experiment.
    n = len(df_raw)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[perm[: 3 * n // 4]] = True

    log.info("feature engineering (Appendix B)...")
    df_eng = engineer_features(df_raw, train_mask)
    log.info(f"engineered {len(df_eng):,} rows × {len(df_eng.columns)-1} features")

    # ── 3. sub-sample for r ────────────────────────────────────────
    log.info(f"sub-sampling for r={args.anomaly_ratio} (n_anomalies={args.n_anomalies})...")
    df_sub = subsample_for_anomaly_ratio(
        df_eng,
        anomaly_ratio=args.anomaly_ratio,
        n_anomalies=args.n_anomalies,
        random_state=args.seed,
    )
    log.info(f"subsampled: {len(df_sub)} rows ({int(df_sub['is_fraud'].sum())} fraud)")

    feature_cols = [c for c in df_sub.columns if c != "is_fraud"]
    X_raw = df_sub[feature_cols].to_numpy()
    y = df_sub["is_fraud"].to_numpy().astype(int)

    # ── 4. classical rbf-OCSVM baseline ────────────────────────────
    log.info("\n=== classical rbf-OCSVM (10-fold CV) ===")
    rbf_results = run_classical_cv(
        X_raw, y, gamma=args.gamma, nu=args.nu, k=args.cv_folds, seed=args.seed
    )
    log.info(f"  train F1 = {rbf_results['train_mean']:.3f} ± {rbf_results['train_std']:.3f}")
    log.info(f"  test  F1 = {rbf_results['test_mean']:.3f} ± {rbf_results['test_std']:.3f}")

    all_results: Dict[str, Dict] = {"rbf": rbf_results}

    # ── 5. quantum qrbf-OCSVM ──────────────────────────────────────
    if not args.skip_quantum:
        # Scale features into [-π, π] for rotation-angle encoding (paper Section III).
        X_scaled = scale_to_pi(X_raw)

        for fm_name in args.feature_maps:
            log.info(f"\n=== qrbf-{fm_name} (10-fold CV) ===")
            q_results = run_quantum_cv(
                X_scaled, y, fm_name,
                gamma=args.gamma, nu=args.nu, k=args.cv_folds, seed=args.seed,
            )
            log.info(f"  train F1 = {q_results['train_mean']:.3f} ± {q_results['train_std']:.3f}")
            log.info(f"  test  F1 = {q_results['test_mean']:.3f} ± {q_results['test_std']:.3f}")
            all_results[f"qrbf-{fm_name}"] = q_results

    # ── 6. compare to paper ────────────────────────────────────────
    log.info("\n=== comparison vs paper Table 1 (r=0.2 test F1) ===")
    log.info(f"  {'model':12s} | {'measured':>16s} | {'paper':>16s} | {'Δ':>6s} | within ±5pp?")
    log.info("  " + "-" * 72)
    pass_count = 0
    eligible = 0
    for model_name, (paper_mu, paper_sd) in PAPER_TABLE_1.items():
        if model_name not in all_results:
            continue
        eligible += 1
        meas = all_results[model_name]
        delta = meas["test_mean"] - paper_mu
        within = abs(delta) <= TOLERANCE
        if within:
            pass_count += 1
        flag = "✓" if within else "✗"
        log.info(
            f"  {model_name:12s} | {meas['test_mean']:.3f} ± {meas['test_std']:.3f} "
            f"| {paper_mu:.3f} ± {paper_sd:.3f} | {delta:+.3f} | {flag}"
        )

    summary = {
        "args": vars(args),
        "results": all_results,
        "paper_targets": PAPER_TABLE_1,
        "tolerance_pp": TOLERANCE,
        "n_within_tolerance": pass_count,
        "n_eligible": eligible,
        "overall_pass": pass_count == eligible,
    }
    out_path = RESULTS_DIR / "results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    log.info(f"\nresults written to {out_path}")

    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
