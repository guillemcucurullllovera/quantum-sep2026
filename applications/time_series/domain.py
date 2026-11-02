"""Typed domain models for the time-series / volatility application layer.

Mirrors the style of applications.portfolio.domain: callers build a
TimeSeriesForecastSpec (plus a FinancialDataset) instead of hand-assembling
QuantumReservoirConfig/ESNConfig objects and wiring ridge regression
themselves. This is the intended user-facing entry point; the underlying
algorithm configs (QuantumReservoirConfig, ESNConfig, ...) remain the wire
format each backend consumes internally.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from core.schemas import BenchmarkResult


class TargetKind(str, Enum):
    REALIZED_VOLATILITY = "realized_volatility"
    RETURN = "return"
    GENERIC = "generic"


class MethodChoice(str, Enum):
    """Every method the unified dispatcher can route a forecast through.

    Four QRC variants are deliberately distinct code paths, not the same
    algorithm under different names:
      - QRC_GEN2: the canonical, current implementation (persistent-state
        architecture, spectral-radius-normalized couplings).
      - QRC_GEN2_ORIGINAL: the preserved pre-refactor implementation
        (algorithms/qrc/gen2_v1_original) -- same math, different code path,
        useful as a parity check on the Gen2 refactor.
      - QRC_GEN1_ARXIV: the paper-faithful replication
        (algorithms_extended.qrc_arxiv_replication) -- windowed architecture,
        UNnormalized couplings, and a documented, honest gate failure (see
        that module's THIRD_PARTY_NOTICES.md). Included and reported as-is, not
        hidden or softened.
      - QRC_GEN1_INFORMED: an ablation this platform adds -- Gen2's
        persistent-state architecture with Gen1's unnormalized couplings.
        Directly answers the open question `algorithms/qrc/reservoir.py`'s
        own docstring raises and leaves untested: does the spectral-radius
        normalization fix matter in isolation from the architecture change?
    """

    QRC_GEN2 = "qrc_gen2"
    QRC_GEN2_ORIGINAL = "qrc_gen2_original"
    QRC_GEN1_ARXIV = "qrc_gen1_arxiv"
    QRC_GEN1_INFORMED = "qrc_gen1_informed"
    ESN = "esn"
    MAMBA_SSM = "mamba_ssm"
    NAIVE_CONSTANT = "naive_constant"
    NAIVE_PERSISTENCE = "naive_persistence"


QUANTUM_METHODS: frozenset[MethodChoice] = frozenset(
    {
        MethodChoice.QRC_GEN2,
        MethodChoice.QRC_GEN2_ORIGINAL,
        MethodChoice.QRC_GEN1_ARXIV,
        MethodChoice.QRC_GEN1_INFORMED,
    }
)
CLASSICAL_METHODS: frozenset[MethodChoice] = frozenset(set(MethodChoice) - QUANTUM_METHODS)
NAIVE_METHODS: frozenset[MethodChoice] = frozenset({MethodChoice.NAIVE_CONSTANT, MethodChoice.NAIVE_PERSISTENCE})


class ValidationStrategy(str, Enum):
    SINGLE_SPLIT = "single_split"
    WALK_FORWARD = "walk_forward"


class FeatureSpec(BaseModel):
    """Preprocessing knobs, wired to core.feature_engineering.fracdiff."""

    use_fracdiff: bool = False
    fracdiff_d: float = 0.4
    fracdiff_threshold: float = 1e-4

    @field_validator("fracdiff_d")
    @classmethod
    def _check_d(cls, v: float) -> float:
        if not 0 < v < 2:
            raise ValueError(f"fracdiff_d must be in (0, 2); got {v}")
        return v

    @field_validator("fracdiff_threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"fracdiff_threshold must be > 0; got {v}")
        return v


class NoiseSpec(BaseModel):
    """Parameters for the noisy-vs-noiseless QRC diagnostic, in the same
    vocabulary hardware.noise.models.synthetic() uses."""

    enabled: bool = False
    depolarizing_prob_1q: float = 0.001
    depolarizing_prob_2q: float = 0.01
    readout_error_prob: float = 0.02

    @field_validator("depolarizing_prob_1q", "depolarizing_prob_2q", "readout_error_prob")
    @classmethod
    def _check_prob(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"probability must be in [0, 1]; got {v}")
        return v


class QRCHyperparams(BaseModel):
    n1: int = 5
    n2: int = 5
    alpha: float = 0.9
    tau: float = 1.0
    field_nu: float = 1.0
    arxiv_window: int = 3  # window length for the windowed Gen1 architecture


class ESNHyperparams(BaseModel):
    reservoir_size: int = 100
    spectral_radius: float = 0.9
    leak_rate: float = 0.3
    input_scaling: float = 1.0
    density: float = 0.1


class MambaHyperparams(BaseModel):
    d_state: int = 16
    expand: int = 2
    n_layers: int = 1


class TimeSeriesForecastSpec(BaseModel):
    """One typed spec fully describing a forecasting run: what target, over
    what horizon, with which preprocessing, evaluated how, comparing which
    methods, under which per-method hyperparameters."""

    target_kind: TargetKind = TargetKind.GENERIC
    horizon: int = 1
    methods: list[MethodChoice] = Field(default_factory=lambda: list(MethodChoice))
    validation: ValidationStrategy = ValidationStrategy.WALK_FORWARD
    n_splits: int = 4
    min_train_fraction: float = 0.5
    test_fraction: float = 0.25  # used only when validation == SINGLE_SPLIT
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    qrc: QRCHyperparams = Field(default_factory=QRCHyperparams)
    esn: ESNHyperparams = Field(default_factory=ESNHyperparams)
    mamba: MambaHyperparams = Field(default_factory=MambaHyperparams)
    ridge_delta: float = 1e-3
    seed: int = 42

    @field_validator("horizon")
    @classmethod
    def _check_horizon(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"horizon must be >= 1; got {v}")
        return v

    @field_validator("n_splits")
    @classmethod
    def _check_n_splits(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_splits must be >= 1; got {v}")
        return v


class FoldMetric(BaseModel):
    fold: int
    n_train: int
    n_test: int
    mse: float
    mae: float


class MethodResult(BaseModel):
    method: MethodChoice
    is_quantum: bool
    fold_metrics: list[FoldMetric] = Field(default_factory=list)
    mean_mse: float
    std_mse: float
    mean_mae: float
    wall_time_s: float
    notes: Optional[str] = None
    failed: bool = False
    error: Optional[str] = None


class NoiseComparisonResult(BaseModel):
    method: MethodChoice
    noiseless_mse: float
    noisy_mse: float
    mse_degradation_ratio: float
    noiseless_mae: float
    noisy_mae: float
    noise_spec: NoiseSpec
    notes: Optional[str] = None


class MultiMethodForecastResult(BaseModel):
    """The one normalized result schema every entry point in this
    application layer (the dispatcher, the CLI, applications.volatility's
    realized-vol specialization) returns."""

    spec: TimeSeriesForecastSpec
    dataset_name: str
    n_samples: int
    n_features: int
    results: dict[str, MethodResult]
    best_method: Optional[str]
    naive_floor_mse: Optional[float]
    arxiv_gate_note: Optional[str] = None
    noise_comparison: Optional[NoiseComparisonResult] = None
    best_quantum_vs_best_classical: Optional[BenchmarkResult] = None
    run_ids: dict[str, str] = Field(default_factory=dict)


__all__ = [
    "TargetKind",
    "MethodChoice",
    "QUANTUM_METHODS",
    "CLASSICAL_METHODS",
    "NAIVE_METHODS",
    "ValidationStrategy",
    "FeatureSpec",
    "NoiseSpec",
    "QRCHyperparams",
    "ESNHyperparams",
    "MambaHyperparams",
    "TimeSeriesForecastSpec",
    "FoldMetric",
    "MethodResult",
    "NoiseComparisonResult",
    "MultiMethodForecastResult",
]
