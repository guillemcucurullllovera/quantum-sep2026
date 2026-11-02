"""P0.1 replication CV harness -- Asymmetrica's own driver code (not from the
paper). Ported from the original prototype's ``run.py`` with two changes:
results now log through ``core.experiment_registry``/``core.result_store``
instead of an ad hoc JSON file, and the Sparkov-specific CLI entrypoint is
kept for exact reproduction of REPLICATION_REPORT.md's numbers.

Algorithm logic (``run_classical_cv``, ``run_quantum_cv``, ``f1_score_anomaly``)
is unchanged from the original -- only I/O plumbing was rewired.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from algorithms_extended.qrbf_ocsvm_anomaly import MEASURED_TEST_F1, PAPER_TABLE_1, TOLERANCE_PP
from algorithms_extended.qrbf_ocsvm_anomaly.kernels import (
    FEATURE_MAPS,
    compute_rdms_for_dataset,
    qrbf_kernel_matrix,
    scale_to_pi,
)
from core.config import get_settings
from core.experiment_registry import ExperimentRegistry
from core.result_store import ResultStore
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata

log = logging.getLogger("qrbf_ocsvm.replication")


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
    """sklearn OneClassSVM returns +1 = inlier, -1 = outlier. Map outlier->1."""
    raw = model.predict(X_or_K)
    return (raw == -1).astype(int)


# ── CV harness ──────────────────────────────────────────────────────

def run_classical_cv(
    X: np.ndarray, y: np.ndarray, gamma: float = 0.1, nu: float = 0.1, k: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """k-fold CV of classical rbf-OCSVM."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train_f1s, test_f1s = [], []

    for tr_idx, te_idx in skf.split(X, y):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
        model.fit(Xtr_s)

        train_f1s.append(f1_score_anomaly(ytr, ocsvm_predict_anomaly(model, Xtr_s)))
        test_f1s.append(f1_score_anomaly(yte, ocsvm_predict_anomaly(model, Xte_s)))

    return {
        "train_mean": float(np.mean(train_f1s)), "train_std": float(np.std(train_f1s)),
        "test_mean": float(np.mean(test_f1s)), "test_std": float(np.std(test_f1s)),
        "per_fold_test": test_f1s,
    }


def run_quantum_cv(
    X_scaled: np.ndarray, y: np.ndarray, feature_map_name: str,
    gamma: float = 0.1, nu: float = 0.1, k: int = 10, seed: int = 42,
    precomputed_rdms: np.ndarray | None = None,
) -> Dict[str, float]:
    """k-fold CV of qrbf-OCSVM for a specific feature map.

    Computes the 1Q-RDM tensor for all samples once (or accepts a
    precomputed tensor, e.g. from the noisy hardware_variant path), then
    builds the kernel matrix per fold.
    """
    if precomputed_rdms is not None:
        rdms = precomputed_rdms
    else:
        fm = FEATURE_MAPS[feature_map_name]
        rdms = compute_rdms_for_dataset(X_scaled, fm)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train_f1s, test_f1s = [], []

    for tr_idx, te_idx in skf.split(X_scaled, y):
        ytr, yte = y[tr_idx], y[te_idx]
        rdms_tr, rdms_te = rdms[tr_idx], rdms[te_idx]

        K_train = qrbf_kernel_matrix(rdms_tr, rdms_tr, gamma=gamma)
        K_test = qrbf_kernel_matrix(rdms_te, rdms_tr, gamma=gamma)

        model = OneClassSVM(kernel="precomputed", nu=nu)
        model.fit(K_train)

        train_f1s.append(f1_score_anomaly(ytr, ocsvm_predict_anomaly(model, K_train)))
        test_f1s.append(f1_score_anomaly(yte, ocsvm_predict_anomaly(model, K_test)))

    return {
        "train_mean": float(np.mean(train_f1s)), "train_std": float(np.std(train_f1s)),
        "test_mean": float(np.mean(test_f1s)), "test_std": float(np.std(test_f1s)),
        "per_fold_test": test_f1s,
    }


def compare_to_paper(all_results: Dict[str, Dict]) -> Dict:
    pass_count = 0
    eligible = 0
    rows = []
    for model_name, (paper_mu, paper_sd) in PAPER_TABLE_1.items():
        if model_name not in all_results:
            continue
        eligible += 1
        meas = all_results[model_name]
        delta = meas["test_mean"] - paper_mu
        within = abs(delta) <= TOLERANCE_PP
        if within:
            pass_count += 1
        rows.append({
            "model": model_name, "measured_mean": meas["test_mean"],
            "measured_std": meas["test_std"], "paper_mean": paper_mu,
            "paper_std": paper_sd, "delta": delta, "within_tolerance": within,
        })
    return {"rows": rows, "n_within_tolerance": pass_count, "n_eligible": eligible,
            "overall_pass": eligible > 0 and pass_count == eligible}


def run_and_record(
    X_raw: np.ndarray, y: np.ndarray, *,
    feature_maps: tuple[str, ...] = ("CX", "ECR", "RXX"),
    gamma: float = 0.1, nu: float = 0.1, k: int = 10, seed: int = 42,
    skip_quantum: bool = False,
) -> ExperimentResult:
    """Run the full classical + quantum CV comparison and log it through
    core.experiment_registry / core.result_store, returning the
    ExperimentResult (same object type every algorithm in this platform uses)."""
    all_results: Dict[str, Dict] = {"rbf": run_classical_cv(X_raw, y, gamma, nu, k, seed)}

    if not skip_quantum:
        X_scaled = scale_to_pi(X_raw)
        for fm_name in feature_maps:
            all_results[f"qrbf-{fm_name}"] = run_quantum_cv(
                X_scaled, y, fm_name, gamma=gamma, nu=nu, k=k, seed=seed
            )

    comparison = compare_to_paper(all_results)

    backend = BackendDescriptor(name="statevector_exact", kind=BackendKind.SIMULATOR, num_qubits=10)
    config = ExperimentConfig(
        name="qrbf_ocsvm_replication", algorithm="qrbf_ocsvm_anomaly", backend=backend,
        seed=seed, params={"gamma": gamma, "nu": nu, "k": k, "feature_maps": list(feature_maps)},
    )
    meta = RunMetadata(
        algorithm="qrbf_ocsvm_anomaly", generation="arxiv_replication",
        adapted_from="Pranjic et al., arXiv:2411.16970", seed=seed,
    )
    metrics = {f"{name}_test_f1": r["test_mean"] for name, r in all_results.items()}
    metrics["n_within_tolerance"] = float(comparison["n_within_tolerance"])
    metrics["n_eligible"] = float(comparison["n_eligible"])

    result = ExperimentResult(
        metadata=meta, config=config, metrics=metrics,
        classical_baseline_metrics={"rbf_test_f1": all_results["rbf"]["test_mean"]},
        notes=(
            "REPLICATION FAILED against paper Table 1 (see REPLICATION_REPORT.md). "
            f"{comparison['n_within_tolerance']}/{comparison['n_eligible']} models within "
            f"+/-{TOLERANCE_PP*100:.0f}pp of the paper's reported F1."
        ),
    )

    store = ResultStore()
    store.save_json(meta.run_id, "full_results", {"per_model": all_results, "comparison": comparison})
    ExperimentRegistry().record(result)
    return result


# ── CLI entrypoint, kept for exact reproduction of REPLICATION_REPORT.md ──

def main(argv=None) -> int:
    from algorithms_extended.qrbf_ocsvm_anomaly.sparkov_data import (
        engineer_features, load_sparkov_raw, subsample_for_anomaly_ratio,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--anomaly-ratio", type=float, default=0.2)
    p.add_argument("--n-anomalies", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--cv-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature-maps", nargs="+", default=["CX", "ECR", "RXX"])
    p.add_argument("--skip-quantum", action="store_true")
    args = p.parse_args(argv)

    workspace_root = Path(get_settings().data_dir).resolve().parents[0] \
        if get_settings().data_dir.exists() else Path.cwd()
    log.info("loading raw Sparkov dataset...")
    df_raw = load_sparkov_raw(workspace_root)

    n = len(df_raw)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[perm[: 3 * n // 4]] = True

    df_eng = engineer_features(df_raw, train_mask)
    df_sub = subsample_for_anomaly_ratio(
        df_eng, anomaly_ratio=args.anomaly_ratio, n_anomalies=args.n_anomalies,
        random_state=args.seed,
    )
    feature_cols = [c for c in df_sub.columns if c != "is_fraud"]
    X_raw = df_sub[feature_cols].to_numpy()
    y = df_sub["is_fraud"].to_numpy().astype(int)

    t0 = time.time()
    result = run_and_record(
        X_raw, y, feature_maps=tuple(args.feature_maps), gamma=args.gamma,
        nu=args.nu, k=args.cv_folds, seed=args.seed, skip_quantum=args.skip_quantum,
    )
    log.info(f"done in {time.time()-t0:.1f}s -- run_id={result.metadata.run_id}")
    log.info(result.notes)
    return 0 if "0/0" not in (result.notes or "") else 1


if __name__ == "__main__":
    sys.exit(main())
