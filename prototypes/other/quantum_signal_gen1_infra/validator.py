"""
Validation utilities for quantum signal vectors.

Provides VectorValidator for checking range constraints and staleness,
plus ValidationReport for structured validation results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime, timezone
from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector


@dataclass
class ValidationReport:
    """
    Structured validation results for a QuantumInputVector.

    Attributes:
        is_valid: True if all checks passed.
        warnings: List of non-critical issues (e.g., unusual values).
        errors: List of critical issues that fail validation.
        checked_at: ISO timestamp when validation was performed.
    """

    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checked_at: str = ""

    def __post_init__(self):
        """Set checked_at to current time if not provided."""
        if not self.checked_at:
            self.checked_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "errors": self.errors,
            "checked_at": self.checked_at,
        }


class VectorValidator:
    """
    Validates QuantumInputVector instances against constraints and staleness.

    Checks:
    - Range constraints on all 14 features
    - Timestamp freshness (data older than threshold is considered stale)
    """

    def __init__(self, stale_threshold_seconds: int = 3600):
        """
        Initialize the validator.

        Args:
            stale_threshold_seconds: Maximum age of data in seconds before
                                    considered stale. Default 1 hour (3600 sec).
        """
        self.stale_threshold_seconds = stale_threshold_seconds

    def validate(self, vector: QuantumInputVector) -> ValidationReport:
        """
        Validate a QuantumInputVector.

        Checks:
        1. All features are within their specified ranges
        2. vol_regime is one of {0, 0.5, 1.0}
        3. Timestamp is not older than stale_threshold

        Args:
            vector: The vector to validate.

        Returns:
            ValidationReport: Detailed results with errors and warnings.
        """
        errors = []
        warnings = []

        # Define range constraints for each feature
        constraints = {
            "momentum_composite": (-1, 1),
            "momentum_acceleration": (-1, 1),
            "realized_vol_20d": (0, 1),
            "vol_regime": None,  # Special handling
            "atr_normalized": (0, 1),
            "carry_differential": (-1, 1),
            "yield_curve_slope": (-1, 1),
            "event_risk_score": (0, 1),
            "net_sentiment": (-1, 1),
            "sentiment_momentum": (-1, 1),
            "cot_positioning": (-1, 1),
            "cot_extreme": (-1, 1),
            "dxy_momentum": (-1, 1),
            "risk_sentiment": (0, 1),
        }

        # Check each feature
        for field_name, bounds in constraints.items():
            value = getattr(vector, field_name)

            if field_name == "vol_regime":
                # vol_regime must be one of {0, 0.5, 1.0}
                if value not in (0, 0.5, 1.0):
                    errors.append(
                        f"{field_name}: must be 0, 0.5, or 1.0, got {value}"
                    )
            else:
                min_val, max_val = bounds
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"{field_name}: out of range [{min_val}, {max_val}], got {value}"
                    )

        # Check timestamp staleness
        try:
            # Try ISO 8601 format parsing
            if 'Z' in vector.timestamp:
                vector_dt = datetime.fromisoformat(vector.timestamp.replace('Z', '+00:00'))
            else:
                vector_dt = datetime.fromisoformat(vector.timestamp)

            # Use aware datetime for comparison
            from datetime import timezone
            now = datetime.now(timezone.utc)

            # If vector_dt is naive, assume UTC
            if vector_dt.tzinfo is None:
                vector_dt = vector_dt.replace(tzinfo=timezone.utc)

            age_seconds = (now - vector_dt).total_seconds()

            if age_seconds > self.stale_threshold_seconds:
                warnings.append(
                    f"Data is stale: {age_seconds:.0f}s old "
                    f"(threshold: {self.stale_threshold_seconds}s)"
                )
            elif age_seconds < 0:
                warnings.append(
                    f"Timestamp is in the future: {vector.timestamp}"
                )

        except (ValueError, AttributeError) as e:
            warnings.append(f"Could not parse timestamp: {vector.timestamp} ({e})")

        # Determine overall validity
        is_valid = len(errors) == 0

        report = ValidationReport(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
        )

        return report

    def validate_batch(self, vectors: List[QuantumInputVector]) -> Dict[str, Any]:
        """
        Validate a batch of vectors and aggregate results.

        Args:
            vectors: List of QuantumInputVectors to validate.

        Returns:
            dict: Aggregated validation results with keys:
            - total: Number of vectors validated
            - valid: Number of vectors with no errors
            - invalid: Number of vectors with errors
            - reports: List of individual ValidationReport objects
            - all_errors: List of all unique error messages across batch
            - all_warnings: List of all unique warning messages across batch
        """
        reports = [self.validate(v) for v in vectors]

        all_errors = set()
        all_warnings = set()

        for report in reports:
            all_errors.update(report.errors)
            all_warnings.update(report.warnings)

        valid_count = sum(1 for r in reports if r.is_valid)
        invalid_count = len(reports) - valid_count

        return {
            "total": len(vectors),
            "valid": valid_count,
            "invalid": invalid_count,
            "reports": reports,
            "all_errors": sorted(list(all_errors)),
            "all_warnings": sorted(list(all_warnings)),
        }
