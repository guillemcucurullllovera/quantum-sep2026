"""Unified dispatcher: ONE entry point that routes a `ClassificationTaskSpec`
to any of the ~16 concrete classification/regression/anomaly implementations
already built across `algorithms/`, `algorithms_extended/`, and
`classical/` in Waves 1-4, and returns a normalized `DispatchResult`.

This is the single most valuable thing this platform adds: before it, comparing
two of these implementations meant reading two different modules' bespoke
dataclasses and writing glue by hand (see `three_way_comparison.py`,
`kernel_classifier.py`, `finance_extensions.classifier_comparison`, each of
which hand-wires 2-4 methods together for one specific comparison). Now any
caller -- a rolling evaluation, a benchmark report, the CLI, a regime
ensemble -- asks for one of `Algorithm`'s ~16 members and gets back the same
`DispatchResult` shape regardless of which underlying library, gradient
method, or label convention that algorithm actually uses internally.

Routing table (`Algorithm` -> what it wraps):
    VQC_V1_LINEAR / VQC_V4_RING / VQC_V5_REUPLOAD
        algorithms.vqc_qml.training.VQCTrainer (ridge-closed-form readout,
        gradient-free outer loop) over the matching algorithms.vqc_qml
        .ansatz.VQCVersion. BINARY + MULTICLASS/REGIME (RidgeReadout is
        already one-vs-rest).
    PQK_GEN1_FIDELITY
        algorithms.pqk.kernels.fidelity_gram (Gen1 8-qubit fixed fidelity
        kernel, historical) + algorithms.pqk.kernels_gen1.SimpleSVM. BINARY
        only (SimpleSVM is a binary SVM).
    PQK_GEN2
        algorithms.pqk.pqk_v2.PQKv2 (KTA-aligned encoding + Nystrom + ridge
        readout). BINARY + MULTICLASS/REGIME.
    QUANTUM_SVC
        algorithms_extended.qml_financial_classifiers.qsvc_pipeline
        .QuantumSVC (precomputed quantum kernel + WeightedSimpleSVM). BINARY.
    FINANCIAL_QNN
        algorithms_extended.qiskit_ml_finance: QNNClassifier for BINARY
        (exact-gradient training via parameter-shift), QNNRegressor for
        REGRESSION -- picked by `spec.task_type`.
    PEGASOS_QSVC
        algorithms_extended.qiskit_ml_asymmetrica.classifiers.pegasos_qsvc
        .PegasosQSVC over a real qiskit ZZFeatureMap + FidelityQuantumKernel.
        BINARY.
    ESTIMATOR_QNN
        algorithms_extended.qiskit_ml_asymmetrica.neural_networks
        .estimator_qnn.EstimatorQNN, wrapped by NeuralNetworkClassifier for
        BINARY or by regressors.vqr.VQR for REGRESSION.
    COVARIANT_KERNEL_SVM
        algorithms_extended.ibm_quantum_kernel_training: CovariantFeatureMap
        trained via SPSA-based KTA alignment (qkt_trainer.spsa_kta_train),
        then a fidelity-kernel SimpleSVM. BINARY.
    REGIME_HMM
        classical.meta_agent.regime_hmm.RegimeHMM (unsupervised 3-state
        Gaussian HMM), scored via Hungarian best-matching against ground
        truth (same technique as `regime_detection.hmm_baseline_comparison`).
        REGIME/MULTICLASS, exactly 3 classes.
    OCSVM_ANOMALY
        classical.meta_agent.ocsvm_anomaly.OCSVMAnomalyDetector. ANOMALY.
    QRBF_OCSVM
        algorithms_extended.qrbf_ocsvm_anomaly (1Q-RDM projected quantum
        kernel + precomputed-kernel OneClassSVM). ANOMALY.
    CLASSICAL_LOGISTIC_REGRESSION / CLASSICAL_RANDOM_FOREST
        classical.machine_learning.baselines. BINARY + MULTICLASS/REGIME.
    CLASSICAL_SVM_RBF
        classical.kernels.baselines' RBF/poly/linear kernels + SimpleSVM.
        BINARY.
    CLASSICAL_RIDGE_REGRESSION
        algorithms_extended.qiskit_ml_finance.classical_baseline
        .ridge_regression_baseline. REGRESSION.

Every adapter function below has the identical signature
`(spec, X_train, y_train, X_test, y_test) -> DispatchResult` and is looked up
through `_ADAPTERS`; `dispatch()` validates task/algorithm compatibility via
`_SUPPORTED_TASKS` before calling it, so an unsupported combination fails
with a clear `UnsupportedTaskError` instead of a confusing internal
exception three layers down.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from algorithms.vqc_qml.ansatz import VQCVersion
from core.metrics import accuracy, calibration_error, mae, mse, rmse
from core.seeds import seed_everything

from applications.classification.errors import AlgorithmUnavailableError, UnsupportedTaskError
from applications.classification.metrics_ext import (
    macro_precision_recall_f1,
    precision_recall_f1,
    roc_auc_binary,
)
from applications.classification.result import DispatchResult
from applications.classification.task_spec import Algorithm, ClassificationTaskSpec, TaskType

_BINARY = {TaskType.BINARY}
_MULTI = {TaskType.MULTICLASS, TaskType.REGIME}
_REG = {TaskType.REGRESSION}
_ANOM = {TaskType.ANOMALY}

_SUPPORTED_TASKS: dict[Algorithm, set[TaskType]] = {
    Algorithm.VQC_V1_LINEAR: _BINARY | _MULTI,
    Algorithm.VQC_V4_RING: _BINARY | _MULTI,
    Algorithm.VQC_V5_REUPLOAD: _BINARY | _MULTI,
    Algorithm.PQK_GEN1_FIDELITY: set(_BINARY),
    Algorithm.PQK_GEN2: _BINARY | _MULTI,
    Algorithm.QUANTUM_SVC: set(_BINARY),
    Algorithm.FINANCIAL_QNN: _BINARY | _REG,
    Algorithm.PEGASOS_QSVC: set(_BINARY),
    Algorithm.ESTIMATOR_QNN: _BINARY | _REG,
    Algorithm.COVARIANT_KERNEL_SVM: set(_BINARY),
    Algorithm.REGIME_HMM: set(_MULTI),
    Algorithm.OCSVM_ANOMALY: set(_ANOM),
    Algorithm.QRBF_OCSVM: set(_ANOM),
    Algorithm.CLASSICAL_LOGISTIC_REGRESSION: _BINARY | _MULTI,
    Algorithm.CLASSICAL_RANDOM_FOREST: _BINARY | _MULTI,
    Algorithm.CLASSICAL_SVM_RBF: set(_BINARY),
    Algorithm.CLASSICAL_RIDGE_REGRESSION: set(_REG),
}


def supported_task_types(algorithm: Algorithm) -> set[TaskType]:
    return set(_SUPPORTED_TASKS[algorithm])


def default_metric_for(task_type: TaskType) -> str:
    return {
        TaskType.BINARY: "accuracy",
        TaskType.MULTICLASS: "accuracy",
        TaskType.REGIME: "accuracy",
        TaskType.REGRESSION: "rmse",
        TaskType.ANOMALY: "f1",
    }[task_type]


# ── shared helpers ───────────────────────────────────────────────────────


def _label_encoder(y_train: np.ndarray):
    """Binary label normalization: returns (to_pm1, from_pm1, neg, pos)
    where `pos`/`neg` are the actual class values seen in y_train (any
    dtype), so every ±1-only algorithm (QuantumSVC, classical SVM/RBF,
    covariant-kernel SVM, EstimatorQNN classifier) can be fed arbitrary
    binary labels (0/1, -1/+1, or anything else) through the same code
    path, and predictions are mapped back to the caller's original labels."""
    classes = np.unique(y_train)
    if len(classes) != 2:
        raise UnsupportedTaskError(
            f"binary algorithm requires exactly 2 classes in y_train, got {len(classes)}: {classes}"
        )
    neg, pos = classes[0], classes[1]

    def to_pm1(y: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(y) == pos, 1.0, -1.0)

    def from_pm1(s: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(s) >= 0, pos, neg)

    return to_pm1, from_pm1, neg, pos


def _binary_metrics(
    y_test: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    calibrate: bool = False,
) -> tuple[dict[str, float], float | None]:
    classes = np.unique(y_test)
    pos = classes[-1] if len(classes) else 1
    metrics = {"accuracy": accuracy(y_test, preds)}
    p, r, f1 = precision_recall_f1(y_test, preds, pos)
    metrics.update(precision=p, recall=r, f1=f1)
    if scores is not None:
        metrics["auc"] = roc_auc_binary(y_test, scores, pos)
    cal_err = None
    if calibrate and probs is not None:
        y_bin = (y_test == pos).astype(float)
        cal_err = calibration_error(np.asarray(probs, dtype=float), y_bin)
        metrics["calibration_error"] = cal_err
    return metrics, cal_err


def _multiclass_metrics(y_test: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    p, r, f1 = macro_precision_recall_f1(y_test, preds)
    return {"accuracy": accuracy(y_test, preds), "macro_precision": p, "macro_recall": r, "macro_f1": f1}


def _regression_metrics(y_test: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {"mse": mse(y_test, preds), "rmse": rmse(y_test, preds), "mae": mae(y_test, preds)}


def _anomaly_metrics(y_test: np.ndarray, preds_anomaly: np.ndarray) -> dict[str, float]:
    p, r, f1 = precision_recall_f1(y_test, preds_anomaly, 1)
    return {"accuracy": accuracy(y_test, preds_anomaly), "precision": p, "recall": r, "f1": f1}


def _result(
    spec: ClassificationTaskSpec,
    n_train: int,
    n_test: int,
    metrics: dict[str, float],
    predictions: np.ndarray,
    **kwargs,
) -> DispatchResult:
    return DispatchResult(
        algorithm=spec.algorithm,
        task_type=spec.task_type,
        n_train=n_train,
        n_test=n_test,
        metrics=metrics,
        predictions=np.asarray(predictions),
        **kwargs,
    )


# ── VQC (v1/v4/v5) ───────────────────────────────────────────────────────


def _make_vqc_adapter(version) -> Callable:
    def adapter(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
        from algorithms.vqc_qml.ansatz import VQCAnsatz
        from algorithms.vqc_qml.calibration import PlattCalibrator
        from algorithms.vqc_qml.training import TrainingConfig, VQCTrainer

        ansatz = VQCAnsatz(n_qubits=spec.n_qubits, version=version)
        cfg = TrainingConfig(
            seed=spec.seed,
            class_weighted=spec.class_weighted,
            max_iter=spec.algorithm_params.get("max_iter", 150),
        )
        trainer = VQCTrainer(ansatz, cfg).fit(X_train, y_train)
        preds = trainer.predict(X_test)
        extra = {"final_train_loss": trainer.loss_history_[-1] if trainer.loss_history_ else None}

        if spec.task_type == TaskType.BINARY:
            classes = list(trainer.readout_.classes_)
            pos_idx = classes.index(np.unique(y_train)[-1])
            scores = trainer.decision_scores(X_test)[:, pos_idx]
            probs = None
            if spec.calibrate:
                train_scores = trainer.decision_scores(X_train)[:, pos_idx]
                y_train_bin = (y_train == np.unique(y_train)[-1]).astype(float)
                calibrator = PlattCalibrator().fit(train_scores, y_train_bin)
                probs = calibrator.predict_proba(scores)
            metrics, cal_err = _binary_metrics(y_test, preds, scores=scores, probs=probs, calibrate=spec.calibrate)
            return _result(
                spec, len(y_train), len(y_test), metrics, preds,
                probabilities=probs, decision_scores=scores, calibration_error=cal_err, extra=extra,
            )

        metrics = _multiclass_metrics(y_test, preds)
        return _result(spec, len(y_train), len(y_test), metrics, preds, extra=extra)

    return adapter


# ── PQK Gen1 (fixed 8-qubit fidelity kernel, historical) ────────────────


def _adapt_pqk_gen1(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from algorithms.pqk.kernels import fidelity_gram
    from algorithms.pqk.kernels_gen1 import SimpleSVM
    from algorithms.vqc_qml.encoding import truncate_or_pad

    # Gen1's feature maps hardcode an 8-qubit circuit (see feature_maps.py);
    # spec.n_qubits does not control Gen1's width -- documented, not silent.
    n_gen1 = 8
    Xtr = truncate_or_pad(X_train, n_gen1)
    Xte = truncate_or_pad(X_test, n_gen1)
    feature_map_name = spec.algorithm_params.get("feature_map", "AngleCNOT")

    to_pm1, from_pm1, _, _ = _label_encoder(y_train)
    K_train = fidelity_gram(Xtr, feature_map_name)
    K_test = fidelity_gram(Xte, feature_map_name, Xtr, psd_correct=False)

    svm = SimpleSVM(C=spec.algorithm_params.get("C", 1.0))
    svm.fit(K_train, to_pm1(y_train))
    scores = svm.decision_function(K_test)
    preds = from_pm1(np.sign(scores))
    metrics, _ = _binary_metrics(y_test, preds, scores=scores)
    return _result(spec, len(y_train), len(y_test), metrics, preds, decision_scores=scores,
                    extra={"feature_map": feature_map_name, "n_qubits_used": n_gen1})


# ── PQK Gen2 (KTA-aligned, Nystrom) ───────────────────────────────────────


def _adapt_pqk_gen2(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from algorithms.pqk.pqk_v2 import PQKConfig, PQKv2

    cfg = PQKConfig(
        n_qubits=spec.n_qubits,
        seed=spec.seed,
        kta_steps=spec.algorithm_params.get("kta_steps", 20),
        nystrom_m=min(spec.algorithm_params.get("nystrom_m", 30), len(X_train)),
    )
    model = PQKv2(cfg).fit(X_train, y_train)
    preds = model.predict(X_test)
    extra = {"final_kta": model.final_kta_, "level1_passed": model.level1_passed}

    if spec.task_type == TaskType.BINARY:
        probs_all = model.predict_proba(X_test)
        classes = list(model.classes_)
        pos_idx = classes.index(np.unique(y_train)[-1])
        scores = probs_all[:, pos_idx]
        metrics, cal_err = _binary_metrics(
            y_test, preds, scores=scores, probs=scores if spec.calibrate else None, calibrate=spec.calibrate
        )
        return _result(
            spec, len(y_train), len(y_test), metrics, preds,
            probabilities=probs_all, decision_scores=scores, calibration_error=cal_err, extra=extra,
        )

    metrics = _multiclass_metrics(y_test, preds)
    return _result(spec, len(y_train), len(y_test), metrics, preds, extra=extra)


# ── QuantumSVC (precomputed-kernel SVM) ───────────────────────────


def _adapt_quantum_svc(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from algorithms_extended.qml_financial_classifiers.qsvc_pipeline import QuantumSVC

    to_pm1, from_pm1, _, _ = _label_encoder(y_train)
    model = QuantumSVC(
        feature_map=spec.algorithm_params.get("feature_map", "AngleCNOTRingV2"),
        kernel_kind=spec.algorithm_params.get("kernel_kind", "fidelity"),
        C=spec.algorithm_params.get("C", 1.0),
        class_weight="balanced" if spec.class_weighted else None,
    )
    model.fit(X_train, to_pm1(y_train))
    K_test = model._gram(X_test, model._X_train)
    scores = model._svm.decision_function(K_test)
    preds = from_pm1(np.sign(scores))
    metrics, _ = _binary_metrics(y_test, preds, scores=scores)
    return _result(spec, len(y_train), len(y_test), metrics, preds, decision_scores=scores)


# ── FinancialQNN (exact-gradient classifier / regressor) ────────────────


def _adapt_financial_qnn(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    if spec.task_type == TaskType.BINARY:
        from algorithms_extended.qiskit_ml_finance.classifier import ClassifierConfig, QNNClassifier

        cfg = ClassifierConfig(
            n_qubits=spec.n_qubits,
            seed=spec.seed,
            epochs=spec.algorithm_params.get("epochs", 40),
            lr=spec.algorithm_params.get("lr", 0.1),
            class_weighted=spec.class_weighted,
        )
        model = QNNClassifier(cfg).fit(X_train, y_train)
        preds = model.predict(X_test)
        scores = model.decision_scores(X_test)
        probs = model.predict_proba(X_test)
        cal_err = None
        if spec.calibrate:
            pos = model.classes_[1]
            y_bin = (y_test == pos).astype(float)
            cal_err = calibration_error(probs, y_bin)
        metrics, _ = _binary_metrics(y_test, preds, scores=scores)
        if cal_err is not None:
            metrics["calibration_error"] = cal_err
        return _result(
            spec, len(y_train), len(y_test), metrics, preds,
            probabilities=probs, decision_scores=scores, calibration_error=cal_err,
            extra={"final_train_loss": model.loss_history_[-1] if model.loss_history_ else None},
        )

    from algorithms_extended.qiskit_ml_finance.regressor import QNNRegressor, RegressorConfig

    cfg = RegressorConfig(
        n_qubits=spec.n_qubits,
        seed=spec.seed,
        epochs=spec.algorithm_params.get("epochs", 40),
        lr=spec.algorithm_params.get("lr", 0.05),
    )
    model = QNNRegressor(cfg).fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = _regression_metrics(y_test, preds)
    return _result(
        spec, len(y_train), len(y_test), metrics, preds,
        extra={"final_train_loss": model.loss_history_[-1] if model.loss_history_ else None},
    )


# ── PegasosQSVC (upstream, real ZZFeatureMap + FidelityQuantumKernel) ───


def _adapt_pegasos(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from qiskit.circuit.library import ZZFeatureMap

    from algorithms.vqc_qml.encoding import truncate_or_pad
    from algorithms_extended.qiskit_ml_asymmetrica.classifiers.pegasos_qsvc import PegasosQSVC
    from algorithms_extended.qiskit_ml_asymmetrica.kernels.fidelity_quantum_kernel import FidelityQuantumKernel

    Xtr = truncate_or_pad(X_train, spec.n_qubits)
    Xte = truncate_or_pad(X_test, spec.n_qubits)
    fm = ZZFeatureMap(spec.n_qubits, reps=spec.algorithm_params.get("reps", 1))
    kernel = FidelityQuantumKernel(feature_map=fm)
    model = PegasosQSVC(
        quantum_kernel=kernel,
        C=spec.algorithm_params.get("C", 1.0),
        num_steps=spec.algorithm_params.get("num_steps", 100),
        seed=spec.seed,
    )
    model.fit(Xtr, np.asarray(y_train))
    preds = model.predict(Xte)
    scores = model.decision_function(Xte)
    probs = model.predict_proba(Xte)[:, 1]
    metrics, cal_err = _binary_metrics(y_test, preds, scores=scores, probs=probs, calibrate=spec.calibrate)
    return _result(
        spec, len(y_train), len(y_test), metrics, preds,
        probabilities=probs, decision_scores=scores, calibration_error=cal_err,
    )


# ── EstimatorQNN (classifier via squared-error sign, VQR for regression) ─


def _adapt_estimator_qnn(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

    from algorithms.vqc_qml.encoding import truncate_or_pad
    from algorithms_extended.qiskit_ml_asymmetrica.neural_networks.estimator_qnn import EstimatorQNN
    from algorithms_extended.qiskit_ml_asymmetrica.optimizers.spsa import SPSA

    n_qubits = spec.n_qubits
    Xtr = truncate_or_pad(X_train, n_qubits)
    Xte = truncate_or_pad(X_test, n_qubits)
    fm = ZZFeatureMap(n_qubits, reps=spec.algorithm_params.get("reps", 1))
    ansatz = RealAmplitudes(n_qubits, reps=spec.algorithm_params.get("ansatz_reps", 1))
    circuit = fm.compose(ansatz)
    maxiter = spec.algorithm_params.get("maxiter", 40)

    if spec.task_type == TaskType.BINARY:
        from algorithms_extended.qiskit_ml_asymmetrica.classifiers.neural_network_classifier import (
            NeuralNetworkClassifier,
        )

        qnn = EstimatorQNN(circuit=circuit, input_params=fm.parameters, weight_params=ansatz.parameters)
        to_pm1, from_pm1, _, _ = _label_encoder(y_train)
        clf = NeuralNetworkClassifier(neural_network=qnn, loss="squared_error", optimizer=SPSA(maxiter=maxiter))
        clf.fit(Xtr, to_pm1(y_train))
        preds_pm1 = clf.predict(Xte)
        preds = from_pm1(preds_pm1)
        scores = np.asarray(clf.neural_network.forward(Xte, clf._fit_result.x)).reshape(-1)
        metrics, _ = _binary_metrics(y_test, preds, scores=scores)
        return _result(spec, len(y_train), len(y_test), metrics, preds, decision_scores=scores)

    from algorithms_extended.qiskit_ml_asymmetrica.regressors.vqr import VQR

    vqr = VQR(feature_map=fm, ansatz=ansatz, optimizer=SPSA(maxiter=maxiter))
    vqr.fit(Xtr, np.asarray(y_train, dtype=float))
    preds = np.asarray(vqr.predict(Xte)).reshape(-1)
    metrics = _regression_metrics(y_test, preds)
    return _result(spec, len(y_train), len(y_test), metrics, preds)


# ── Covariant kernel (SPSA-trained CovariantFeatureMap + fidelity SVM) ──


def _adapt_covariant_kernel(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from qiskit_aer import AerSimulator

    from algorithms.pqk.kernels_gen1 import SimpleSVM
    from algorithms.vqc_qml.encoding import truncate_or_pad
    from algorithms_extended.ibm_quantum_kernel_training.covariant_feature_map import CovariantFeatureMap
    from algorithms_extended.ibm_quantum_kernel_training.qkt_trainer import fidelity_kernel_entry, spsa_kta_train

    n_qubits = spec.n_qubits
    feat_dim = 2 * n_qubits
    Xtr = truncate_or_pad(X_train, feat_dim)
    Xte = truncate_or_pad(X_test, feat_dim)
    fm = CovariantFeatureMap(feature_dimension=feat_dim)

    to_pm1, from_pm1, _, _ = _label_encoder(y_train)
    rng = np.random.default_rng(spec.seed)
    theta0 = rng.uniform(0, 2 * np.pi, n_qubits)
    maxiter = spec.algorithm_params.get("maxiter", 5)
    shots = spec.algorithm_params.get("shots", 64)

    backend = AerSimulator()
    best_theta, best_kta, history = spsa_kta_train(
        fm, Xtr, to_pm1(y_train), theta0, maxiter=maxiter, backend=backend, shots=shots
    )

    def kfn(x1: np.ndarray, x2: np.ndarray) -> float:
        return fidelity_kernel_entry(fm, x1, x2, best_theta, backend=backend, shots=shots)

    K_train = np.array([[kfn(a, b) for b in Xtr] for a in Xtr])
    K_test = np.array([[kfn(a, b) for b in Xtr] for a in Xte])

    svm = SimpleSVM(C=spec.algorithm_params.get("C", 1.0))
    svm.fit(K_train, to_pm1(y_train))
    scores = svm.decision_function(K_test)
    preds = from_pm1(np.sign(scores))
    metrics, _ = _binary_metrics(y_test, preds, scores=scores)
    return _result(
        spec, len(y_train), len(y_test), metrics, preds, decision_scores=scores,
        extra={"final_kta": best_kta, "kta_history": history},
    )


# ── RegimeHMM (unsupervised, Hungarian-matched) ──────────────────────────


def _adapt_regime_hmm(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from scipy.optimize import linear_sum_assignment

    from classical.meta_agent.regime_hmm import RegimeHMM

    classes = np.unique(y_train)
    if len(classes) != 3:
        raise UnsupportedTaskError(
            f"REGIME_HMM requires exactly 3 classes/regimes (bull/bear/crisis); got {len(classes)}"
        )
    vix_col = spec.algorithm_params.get("vix_col", 0)
    hmm = RegimeHMM(n_states=3, vix_col=vix_col, random_state=spec.seed)
    hmm.fit(X_train)
    cluster_labels = hmm.predict(X_test)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test_idx = np.array([class_to_idx[v] for v in y_test])
    confusion = np.zeros((3, 3), dtype=int)
    for t, c in zip(y_test_idx, cluster_labels):
        confusion[t, c] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    cluster_to_true_idx = {int(col): int(row) for row, col in zip(row_ind, col_ind)}
    preds_idx = np.array([cluster_to_true_idx.get(int(c), c) for c in cluster_labels])
    preds = classes[preds_idx]

    metrics = _multiclass_metrics(y_test, preds)
    summary = hmm.summary()
    return _result(
        spec, len(y_train), len(y_test), metrics, preds,
        extra={"transmat": summary.transmat, "state_labels": summary.state_labels},
    )


# ── OCSVM anomaly (classical) ─────────────────────────────────────────────


def _adapt_ocsvm_anomaly(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from classical.meta_agent.ocsvm_anomaly import OCSVMAnomalyDetector

    nu = spec.algorithm_params.get("nu", 0.1)
    gamma = spec.algorithm_params.get("gamma", "scale")
    det = OCSVMAnomalyDetector(nu=nu, gamma=gamma, random_state=spec.seed).fit(X_train)
    preds = det.predict_anomalous(X_test).astype(int)
    scores = det.anomaly_score(X_test)
    metrics = _anomaly_metrics(y_test, preds)
    metrics["auc"] = roc_auc_binary(y_test, scores, 1)
    return _result(spec, len(y_train), len(y_test), metrics, preds, decision_scores=scores, extra={"nu": nu})


# ── qrbf-OCSVM anomaly (quantum, documented replication failure) ────────


def _adapt_qrbf_ocsvm(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from sklearn.svm import OneClassSVM

    from algorithms.vqc_qml.encoding import truncate_or_pad
    from algorithms_extended.qrbf_ocsvm_anomaly.kernels import (
        FEATURE_MAPS,
        N_QUBITS,
        compute_rdms_for_dataset,
        qrbf_kernel_matrix,
        scale_to_pi,
    )

    n_feat = 2 * N_QUBITS  # feature_map_{cx,ecr,rxx} require x[k], x[N_QUBITS+k] for k in range(N_QUBITS)
    Xtr = truncate_or_pad(X_train, n_feat)
    Xte = truncate_or_pad(X_test, n_feat)
    feature_map_name = spec.algorithm_params.get("feature_map", "ECR")
    gamma = spec.algorithm_params.get("gamma", 0.1)
    nu = spec.algorithm_params.get("nu", 0.1)

    fm = FEATURE_MAPS[feature_map_name]
    Xtr_s, Xte_s = scale_to_pi(Xtr), scale_to_pi(Xte)
    rdms_tr = compute_rdms_for_dataset(Xtr_s, fm)
    rdms_te = compute_rdms_for_dataset(Xte_s, fm)
    K_train = qrbf_kernel_matrix(rdms_tr, rdms_tr, gamma=gamma)
    K_test = qrbf_kernel_matrix(rdms_te, rdms_tr, gamma=gamma)

    model = OneClassSVM(kernel="precomputed", nu=nu).fit(K_train)
    preds = (model.predict(K_test) == -1).astype(int)
    scores = -model.decision_function(K_test)
    metrics = _anomaly_metrics(y_test, preds)
    metrics["auc"] = roc_auc_binary(y_test, scores, 1)
    return _result(
        spec, len(y_train), len(y_test), metrics, preds, decision_scores=scores,
        extra={"feature_map": feature_map_name},
    )


# ── classical baselines ───────────────────────────────────────────────────


def _adapt_classical_logistic(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from classical.machine_learning.baselines import _HAS_SKLEARN, logistic_regression_baseline

    if spec.task_type in _MULTI and not _HAS_SKLEARN:
        raise AlgorithmUnavailableError("classical_logistic_regression multiclass path requires scikit-learn")
    result = logistic_regression_baseline(X_train, np.asarray(y_train), X_test, np.asarray(y_test))
    if spec.task_type == TaskType.BINARY:
        probs = result.probabilities[:, -1]
        metrics, cal_err = _binary_metrics(y_test, result.predictions, scores=probs, probs=probs, calibrate=spec.calibrate)
        return _result(
            spec, len(y_train), len(y_test), metrics, result.predictions,
            probabilities=result.probabilities, decision_scores=probs, calibration_error=cal_err,
        )
    metrics = _multiclass_metrics(y_test, result.predictions)
    return _result(spec, len(y_train), len(y_test), metrics, result.predictions, probabilities=result.probabilities)


def _adapt_classical_rf(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from classical.machine_learning.baselines import _HAS_SKLEARN, random_forest_baseline

    if not _HAS_SKLEARN:
        raise AlgorithmUnavailableError("classical_random_forest requires scikit-learn")
    result = random_forest_baseline(
        X_train, np.asarray(y_train), X_test, np.asarray(y_test),
        n_estimators=spec.algorithm_params.get("n_estimators", 200),
    )
    if spec.task_type == TaskType.BINARY:
        probs = result.probabilities[:, -1]
        metrics, cal_err = _binary_metrics(y_test, result.predictions, scores=probs, probs=probs, calibrate=spec.calibrate)
        return _result(
            spec, len(y_train), len(y_test), metrics, result.predictions,
            probabilities=result.probabilities, decision_scores=probs, calibration_error=cal_err,
        )
    metrics = _multiclass_metrics(y_test, result.predictions)
    return _result(spec, len(y_train), len(y_test), metrics, result.predictions, probabilities=result.probabilities)


def _adapt_classical_svm_rbf(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from algorithms.pqk.kernels_gen1 import SimpleSVM
    from classical.kernels.baselines import CLASSICAL_KERNELS

    to_pm1, from_pm1, _, _ = _label_encoder(y_train)
    kernel_name = spec.algorithm_params.get("kernel", "rbf")
    kfn = CLASSICAL_KERNELS[kernel_name]
    kwargs = {k: v for k, v in spec.algorithm_params.items() if k in ("gamma", "degree", "coef0")}
    K_train = kfn(X_train, X_train, **kwargs)
    K_test = kfn(X_test, X_train, **kwargs)
    svm = SimpleSVM(C=spec.algorithm_params.get("C", 1.0))
    svm.fit(K_train, to_pm1(y_train))
    scores = svm.decision_function(K_test)
    preds = from_pm1(np.sign(scores))
    metrics, _ = _binary_metrics(y_test, preds, scores=scores)
    return _result(spec, len(y_train), len(y_test), metrics, preds, decision_scores=scores,
                    extra={"kernel": kernel_name})


def _adapt_classical_ridge(spec: ClassificationTaskSpec, X_train, y_train, X_test, y_test) -> DispatchResult:
    from algorithms_extended.qiskit_ml_finance.classical_baseline import ridge_regression_baseline

    result = ridge_regression_baseline(
        X_train, np.asarray(y_train, dtype=float), X_test, np.asarray(y_test, dtype=float),
        alpha=spec.algorithm_params.get("alpha", 1.0),
    )
    metrics = _regression_metrics(y_test, result.predictions)
    return _result(spec, len(y_train), len(y_test), metrics, result.predictions)


# ── registry + entry point ────────────────────────────────────────────────

_ADAPTERS: dict[Algorithm, Callable] = {
    Algorithm.VQC_V1_LINEAR: _make_vqc_adapter(VQCVersion.V1_LINEAR),
    Algorithm.VQC_V4_RING: _make_vqc_adapter(VQCVersion.V4_RING),
    Algorithm.VQC_V5_REUPLOAD: _make_vqc_adapter(VQCVersion.V5_REUPLOAD),
    Algorithm.PQK_GEN1_FIDELITY: _adapt_pqk_gen1,
    Algorithm.PQK_GEN2: _adapt_pqk_gen2,
    Algorithm.QUANTUM_SVC: _adapt_quantum_svc,
    Algorithm.FINANCIAL_QNN: _adapt_financial_qnn,
    Algorithm.PEGASOS_QSVC: _adapt_pegasos,
    Algorithm.ESTIMATOR_QNN: _adapt_estimator_qnn,
    Algorithm.COVARIANT_KERNEL_SVM: _adapt_covariant_kernel,
    Algorithm.REGIME_HMM: _adapt_regime_hmm,
    Algorithm.OCSVM_ANOMALY: _adapt_ocsvm_anomaly,
    Algorithm.QRBF_OCSVM: _adapt_qrbf_ocsvm,
    Algorithm.CLASSICAL_LOGISTIC_REGRESSION: _adapt_classical_logistic,
    Algorithm.CLASSICAL_RANDOM_FOREST: _adapt_classical_rf,
    Algorithm.CLASSICAL_SVM_RBF: _adapt_classical_svm_rbf,
    Algorithm.CLASSICAL_RIDGE_REGRESSION: _adapt_classical_ridge,
}


def dispatch(
    spec: ClassificationTaskSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> DispatchResult:
    """Route `spec` to its algorithm adapter and return a normalized
    `DispatchResult`. Validates task/algorithm compatibility up front."""
    supported = _SUPPORTED_TASKS.get(spec.algorithm)
    if supported is None:
        raise UnsupportedTaskError(f"Unknown algorithm {spec.algorithm!r}")
    if spec.task_type not in supported:
        raise UnsupportedTaskError(
            f"{spec.algorithm.value} does not support task_type={spec.task_type.value}. "
            f"Supported: {sorted(t.value for t in supported)}"
        )
    seed_everything(spec.seed)
    adapter = _ADAPTERS[spec.algorithm]
    t0 = time.perf_counter()
    result = adapter(
        spec,
        np.asarray(X_train, dtype=float),
        np.asarray(y_train),
        np.asarray(X_test, dtype=float),
        np.asarray(y_test),
    )
    result.fit_time_s = time.perf_counter() - t0
    return result


__all__ = ["dispatch", "supported_task_types", "default_metric_for"]
