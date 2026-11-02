"""
Signal history logging and tracking utilities.

Provides SignalHistoryLogger for persisting QuantumInputVectors to CSV files
and retrieving historical data for analysis.
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector


def build_quantum_vector(
    signal_dict: Dict[str, Any],
    asset: str,
    timestamp: str,
) -> QuantumInputVector:
    """
    Factory function to build a QuantumInputVector from raw signal data.

    Combines signal data from multiple agents with metadata to create
    a validated quantum vector. Used by the public API.

    Args:
        signal_dict: Dictionary with the 14 required fields.
        asset: Asset identifier (e.g., 'BTC/USD').
        timestamp: ISO timestamp string.

    Returns:
        QuantumInputVector: Validated quantum vector.

    Raises:
        ValueError: If signal_dict doesn't contain required fields or
                   values are out of range.
    """
    signal_dict["asset"] = asset
    signal_dict["timestamp"] = timestamp
    return QuantumInputVector.from_dict(signal_dict)


class SignalHistoryLogger:
    """
    Logs QuantumInputVectors to a CSV file with append mode.

    Maintains a persistent historical record of all signals processed,
    including market returns computed later.
    """

    def __init__(self, log_file: str):
        """
        Initialize the logger with a CSV file path.

        Args:
            log_file: Path to CSV file. Will be created if it doesn't exist.
        """
        self.log_file = log_file
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create the CSV file with headers if it doesn't exist."""
        if os.path.exists(self.log_file):
            return

        # Create parent directory if needed
        parent_dir = os.path.dirname(self.log_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Write header row
        headers = [
            "timestamp",
            "asset",
            "momentum_composite",
            "momentum_acceleration",
            "realized_vol_20d",
            "vol_regime",
            "atr_normalized",
            "carry_differential",
            "yield_curve_slope",
            "event_risk_score",
            "net_sentiment",
            "sentiment_momentum",
            "cot_positioning",
            "cot_extreme",
            "dxy_momentum",
            "risk_sentiment",
            "market_return_1d",
        ]

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log(
        self,
        vector: QuantumInputVector,
        market_return_1d: Optional[float] = None,
    ) -> None:
        """
        Log a QuantumInputVector to the CSV file.

        Appends one row with the vector's features and optional market return.

        Args:
            vector: The QuantumInputVector to log.
            market_return_1d: Optional 1-day market return (added later). Default None.
        """
        market_return_1d = market_return_1d if market_return_1d is not None else ""

        row = [
            vector.timestamp,
            vector.asset,
            vector.momentum_composite,
            vector.momentum_acceleration,
            vector.realized_vol_20d,
            vector.vol_regime,
            vector.atr_normalized,
            vector.carry_differential,
            vector.yield_curve_slope,
            vector.event_risk_score,
            vector.net_sentiment,
            vector.sentiment_momentum,
            vector.cot_positioning,
            vector.cot_extreme,
            vector.dxy_momentum,
            vector.risk_sentiment,
            market_return_1d,
        ]

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def flush(self) -> None:
        """
        Flush any pending writes (currently a no-op but kept for API consistency).

        In a production system, this would flush file buffers if using buffered I/O.
        """
        pass

    def load_history(self) -> List[Dict[str, Any]]:
        """
        Load all logged signals from the CSV file.

        Reads the entire CSV and returns each row as a dictionary.
        Numeric fields are parsed as floats; market_return_1d may be empty string.

        Returns:
            list[dict]: List of signal dictionaries, one per CSV row (excluding header).

        Raises:
            FileNotFoundError: If log file doesn't exist (call log() first).
        """
        if not os.path.exists(self.log_file):
            raise FileNotFoundError(f"Log file not found: {self.log_file}")

        history = []

        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse numeric fields
                for field in [
                    "momentum_composite", "momentum_acceleration",
                    "realized_vol_20d", "vol_regime", "atr_normalized",
                    "carry_differential", "yield_curve_slope",
                    "event_risk_score", "net_sentiment", "sentiment_momentum",
                    "cot_positioning", "cot_extreme", "dxy_momentum",
                    "risk_sentiment"
                ]:
                    if field in row:
                        try:
                            row[field] = float(row[field])
                        except ValueError:
                            row[field] = None

                # Parse market return if present and non-empty
                if "market_return_1d" in row and row["market_return_1d"]:
                    try:
                        row["market_return_1d"] = float(row["market_return_1d"])
                    except ValueError:
                        row["market_return_1d"] = None
                else:
                    row["market_return_1d"] = None

                history.append(row)

        return history

    def update_market_return(self, timestamp: str, asset: str, return_value: float) -> None:
        """
        Update the market_return_1d field for a specific timestamp/asset pair.

        Reads the entire CSV, updates the matching row, and rewrites the file.

        Args:
            timestamp: Timestamp to match.
            asset: Asset to match.
            return_value: The market return value to set.

        Raises:
            FileNotFoundError: If log file doesn't exist.
            ValueError: If no row matches the timestamp/asset pair.
        """
        if not os.path.exists(self.log_file):
            raise FileNotFoundError(f"Log file not found: {self.log_file}")

        history = self.load_history()
        found = False

        for row in history:
            if row["timestamp"] == timestamp and row["asset"] == asset:
                row["market_return_1d"] = return_value
                found = True
                break

        if not found:
            raise ValueError(f"No entry found for timestamp={timestamp}, asset={asset}")

        # Rewrite the entire file
        headers = [
            "timestamp", "asset",
            "momentum_composite", "momentum_acceleration",
            "realized_vol_20d", "vol_regime", "atr_normalized",
            "carry_differential", "yield_curve_slope", "event_risk_score",
            "net_sentiment", "sentiment_momentum",
            "cot_positioning", "cot_extreme", "dxy_momentum",
            "risk_sentiment", "market_return_1d",
        ]

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(history)
