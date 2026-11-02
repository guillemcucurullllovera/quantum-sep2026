"""
Signal readers for ingesting market data from multiple agent sources.

Supports reading from JSON files, directories, and in-memory dictionaries.
Provides standardized interfaces for technical, macro, news, fundamental,
risk sentiment, and commodity agents.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Union


class SignalReader:
    """Reads raw agent signal outputs from various sources."""

    @staticmethod
    def _validate_fields(data: Dict[str, Any], required_fields: List[str], source_name: str) -> None:
        """
        Validate that required fields are present in data.

        Args:
            data: Dictionary to validate.
            required_fields: List of field names that must be present.
            source_name: Name of the agent/source (for error messages).

        Raises:
            ValueError: If any required fields are missing.
        """
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(
                f"Missing required fields from {source_name}: {missing}"
            )

    @staticmethod
    def _load_json(source: Union[str, Dict]) -> Dict[str, Any]:
        """
        Load JSON from file path or return dict as-is.

        Args:
            source: File path (str) or dictionary.

        Returns:
            dict: Parsed JSON or original dict.

        Raises:
            FileNotFoundError: If file does not exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        if isinstance(source, dict):
            return source

        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found: {source}")
            try:
                with open(source, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Invalid JSON in {source}: {e.msg}", e.doc, e.pos)

        raise TypeError(f"source must be str (file path) or dict, got {type(source)}")

    def read_technical(self, source: Union[str, Dict]) -> Dict[str, Any]:
        """
        Read technical analysis signals from an agent.

        Expected output schema:
        {
            "timestamp": str,
            "asset": str,
            "momentum_composite": float,
            "momentum_acceleration": float,
            "realized_vol_20d": float,
            "vol_regime": float,
            "atr_normalized": float
        }

        Args:
            source: JSON file path or dict.

        Returns:
            dict: Standardized technical signals.

        Raises:
            ValueError: If required fields are missing.
        """
        data = self._load_json(source)
        required = ["timestamp", "asset", "momentum_composite", "momentum_acceleration",
                    "realized_vol_20d", "vol_regime", "atr_normalized"]
        self._validate_fields(data, required, "technical agent")
        return data

    def read_macro(self, source: Union[str, Dict]) -> Dict[str, Any]:
        """
        Read macro economic signals from an agent.

        Expected output schema:
        {
            "timestamp": str,
            "asset": str,
            "carry_differential": float,
            "yield_curve_slope": float,
            "event_risk_score": float
        }

        Args:
            source: JSON file path or dict.

        Returns:
            dict: Standardized macro signals.

        Raises:
            ValueError: If required fields are missing.
        """
        data = self._load_json(source)
        required = ["timestamp", "asset", "carry_differential",
                    "yield_curve_slope", "event_risk_score"]
        self._validate_fields(data, required, "macro agent")
        return data

    def read_news(self, source: Union[str, Dict]) -> Dict[str, Any]:
        """
        Read news sentiment signals from an agent.

        Expected output schema:
        {
            "timestamp": str,
            "asset": str,
            "net_sentiment": float,
            "sentiment_momentum": float
        }

        Args:
            source: JSON file path or dict.

        Returns:
            dict: Standardized news sentiment signals.

        Raises:
            ValueError: If required fields are missing.
        """
        data = self._load_json(source)
        required = ["timestamp", "asset", "net_sentiment", "sentiment_momentum"]
        self._validate_fields(data, required, "news agent")
        return data

    def read_fundamental(self, source: Union[str, Dict]) -> Dict[str, Any]:
        """
        Read fundamental/COT positioning signals from an agent.

        Expected output schema:
        {
            "timestamp": str,
            "asset": str,
            "cot_positioning": float,
            "cot_extreme": float
        }

        Args:
            source: JSON file path or dict.

        Returns:
            dict: Standardized fundamental signals.

        Raises:
            ValueError: If required fields are missing.
        """
        data = self._load_json(source)
        required = ["timestamp", "asset", "cot_positioning", "cot_extreme"]
        self._validate_fields(data, required, "fundamental agent")
        return data

    def read_risk_sentiment(self, source: Union[str, Dict]) -> Dict[str, Any]:
        """
        Read risk sentiment signals from an agent.

        Expected output schema:
        {
            "timestamp": str,
            "asset": str,
            "dxy_momentum": float,
            "risk_sentiment": float
        }

        Args:
            source: JSON file path or dict.

        Returns:
            dict: Standardized risk sentiment signals.

        Raises:
            ValueError: If required fields are missing.
        """
        data = self._load_json(source)
        required = ["timestamp", "asset", "dxy_momentum", "risk_sentiment"]
        self._validate_fields(data, required, "risk sentiment agent")
        return data

    def read_commodity(self, source: Union[str, Dict]) -> Dict[str, Any]:
        """
        Read commodity-related signals from an agent.

        Expected output schema varies by commodity type, but generally:
        {
            "timestamp": str,
            "asset": str,
            ... (commodity-specific fields)
        }

        Args:
            source: JSON file path or dict.

        Returns:
            dict: Commodity signals with timestamp and asset.

        Raises:
            ValueError: If required fields (timestamp, asset) are missing.
        """
        data = self._load_json(source)
        required = ["timestamp", "asset"]
        self._validate_fields(data, required, "commodity agent")
        return data

    def read_all(self, directory: str) -> Dict[str, Any]:
        """
        Read and combine all JSON signal files from a directory.

        Reads all .json files in the directory, assuming each file contains
        one or more signal outputs. Merges all signals into a single dict
        where each agent type is accumulated into a list.

        Args:
            directory: Path to directory containing .json files.

        Returns:
            dict: Combined signals with structure:
            {
                "technical": [list of technical signals],
                "macro": [list of macro signals],
                "news": [list of news signals],
                "fundamental": [list of fundamental signals],
                "risk_sentiment": [list of risk sentiment signals],
                "commodity": [list of commodity signals],
                "raw": [all raw files read]
            }

        Raises:
            FileNotFoundError: If directory does not exist.
            ValueError: If no JSON files found in directory.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        json_files = list(dir_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {directory}")

        combined = {
            "technical": [],
            "macro": [],
            "news": [],
            "fundamental": [],
            "risk_sentiment": [],
            "commodity": [],
            "raw": [],
        }

        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                combined["raw"].append({
                    "file": str(json_file),
                    "data": data
                })

                # Try to classify and add to appropriate list
                # This is heuristic-based on field presence
                if isinstance(data, dict):
                    if "momentum_composite" in data:
                        combined["technical"].append(data)
                    if "carry_differential" in data:
                        combined["macro"].append(data)
                    if "net_sentiment" in data:
                        combined["news"].append(data)
                    if "cot_positioning" in data:
                        combined["fundamental"].append(data)
                    if "dxy_momentum" in data:
                        combined["risk_sentiment"].append(data)

            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON in {json_file}: {e.msg}", e.doc, e.pos
                )

        return combined
