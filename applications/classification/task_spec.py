"""Typed domain models for the unified classification/regime/anomaly product
layer.

Everything the dispatcher (`applications.classification.dispatcher`) needs to
route a run is captured in one `ClassificationTaskSpec`: what kind of problem
this is (`TaskType`), which of the ~16 underlying implementations to run
(`Algorithm`), how many qubits/features to use, how to validate, and any
algorithm-specific knobs (`algorithm_params`) that don't belong in the shared
schema. This replaces the previous pattern of every application module
hardcoding its own algorithm choice and parameter list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.errors import ValidationError


class TaskType(str, Enum):
    """What kind of problem is being solved. Distinct from `Algorithm`
    because several algorithms serve more than one task type (e.g. PQKv2
    handles both BINARY and MULTICLASS; FinancialQNN's classifier/regressor
    split on this field)."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    REGIME = "regime"          # multiclass with a fixed 3-state bull/bear/crisis semantic
    ANOMALY = "anomaly"        # binary, but 1 = anomaly and typically unsupervised at fit time


class Algorithm(str, Enum):
    """Every concrete implementation the unified dispatcher can route to.
    Values are stable strings (used in CLI args, experiment registry
    `algorithm` field, and provenance)."""

    VQC_V1_LINEAR = "vqc_v1_linear"
    VQC_V4_RING = "vqc_v4_ring"
    VQC_V5_REUPLOAD = "vqc_v5_reupload"
    PQK_GEN1_FIDELITY = "pqk_gen1_fidelity"
    PQK_GEN2 = "pqk_gen2"
    QUANTUM_SVC = "quantum_svc"
    FINANCIAL_QNN = "financial_qnn"
    PEGASOS_QSVC = "pegasos_qsvc"
    ESTIMATOR_QNN = "estimator_qnn"
    COVARIANT_KERNEL_SVM = "covariant_kernel_svm"
    REGIME_HMM = "regime_hmm"
    OCSVM_ANOMALY = "ocsvm_anomaly"
    QRBF_OCSVM = "qrbf_ocsvm"
    CLASSICAL_LOGISTIC_REGRESSION = "classical_logistic_regression"
    CLASSICAL_RANDOM_FOREST = "classical_random_forest"
    CLASSICAL_SVM_RBF = "classical_svm_rbf"
    CLASSICAL_RIDGE_REGRESSION = "classical_ridge_regression"


class ValidationStrategy(str, Enum):
    SINGLE_SPLIT = "single_split"
    WALK_FORWARD = "walk_forward"
    ROLLING = "rolling"


QUANTUM_ALGORITHMS = frozenset(
    {
        Algorithm.VQC_V1_LINEAR,
        Algorithm.VQC_V4_RING,
        Algorithm.VQC_V5_REUPLOAD,
        Algorithm.PQK_GEN1_FIDELITY,
        Algorithm.PQK_GEN2,
        Algorithm.QUANTUM_SVC,
        Algorithm.FINANCIAL_QNN,
        Algorithm.PEGASOS_QSVC,
        Algorithm.ESTIMATOR_QNN,
        Algorithm.COVARIANT_KERNEL_SVM,
        Algorithm.QRBF_OCSVM,
    }
)

CLASSICAL_ALGORITHMS = frozenset(
    {
        Algorithm.REGIME_HMM,
        Algorithm.OCSVM_ANOMALY,
        Algorithm.CLASSICAL_LOGISTIC_REGRESSION,
        Algorithm.CLASSICAL_RANDOM_FOREST,
        Algorithm.CLASSICAL_SVM_RBF,
        Algorithm.CLASSICAL_RIDGE_REGRESSION,
    }
)


@dataclass(frozen=True)
class FeatureSpec:
    """What the raw dataset features look like and how they should be
    prepared before hitting an algorithm adapter."""

    n_qubits: int = 4
    fracdiff_d: float | None = None
    fracdiff_threshold: float = 1e-4
    standardize: bool = True

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValidationError("FeatureSpec.n_qubits must be >= 2")
        if self.fracdiff_d is not None and not (0.0 < self.fracdiff_d < 2.0):
            raise ValidationError("FeatureSpec.fracdiff_d must be in (0, 2) when set")


@dataclass
class ClassificationTaskSpec:
    """Everything `dispatcher.dispatch()` needs to run one algorithm on one
    train/test split. This is the single object every application in this
    wave constructs instead of hand-wiring imports."""

    task_type: TaskType
    algorithm: Algorithm
    n_qubits: int = 4
    seed: int = 42
    class_weighted: bool = True
    calibrate: bool = False
    validation_strategy: ValidationStrategy = ValidationStrategy.SINGLE_SPLIT
    n_splits: int = 3
    test_fraction: float = 0.3
    feature_spec: FeatureSpec | None = None
    algorithm_params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValidationError("ClassificationTaskSpec.n_qubits must be >= 2")
        if not (0.0 < self.test_fraction < 1.0):
            raise ValidationError("ClassificationTaskSpec.test_fraction must be in (0, 1)")
        if self.n_splits < 1:
            raise ValidationError("ClassificationTaskSpec.n_splits must be >= 1")
        if self.feature_spec is None:
            self.feature_spec = FeatureSpec(n_qubits=self.n_qubits)

    @property
    def is_quantum(self) -> bool:
        return self.algorithm in QUANTUM_ALGORITHMS
