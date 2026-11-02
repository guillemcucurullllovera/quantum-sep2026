"""
Normalization and scaling utilities for market signals.

Provides stateless transformation functions and a stateful RollingStats
class for computing and maintaining rolling statistics over time series data.
"""

import math
from collections import deque
from typing import List, Dict, Tuple


class Normalizer:
    """Stateless normalization and scaling utilities."""

    @staticmethod
    def standard_scale(
        value: float,
        mean: float,
        std: float,
        clip_range: Tuple[float, float] = (-1, 1),
    ) -> float:
        """
        Standard score (z-score) normalization with optional clipping.

        Transforms: z = (value - mean) / std
        Then clips to clip_range.

        Args:
            value: The value to normalize.
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            clip_range: Tuple (min, max) for clipping. Default (-1, 1).

        Returns:
            float: Normalized and clipped value.

        Raises:
            ValueError: If std is 0 (division by zero).
        """
        if std == 0 or std < 1e-10:
            return 0.0  # No variance — return neutral

        z = (value - mean) / std
        min_clip, max_clip = clip_range
        return max(min_clip, min(max_clip, z))

    @staticmethod
    def percentile_rank(value: float, history: List[float]) -> float:
        """
        Compute percentile rank of value within history.

        Returns the fraction of history values <= value, in [0, 1].

        Args:
            value: The value to rank.
            history: List of historical values.

        Returns:
            float: Percentile rank in [0, 1].

        Raises:
            ValueError: If history is empty.
        """
        if not history:
            raise ValueError("Cannot compute percentile rank on empty history")

        count = sum(1 for x in history if x <= value)
        return count / len(history)

    @staticmethod
    def clip_and_scale(
        value: float,
        input_range: Tuple[float, float],
        output_range: Tuple[float, float],
    ) -> float:
        """
        Clip value to input range, then linearly scale to output range.

        Args:
            value: The value to transform.
            input_range: Tuple (min, max) for input domain.
            output_range: Tuple (min, max) for output domain.

        Returns:
            float: Scaled value in output_range.

        Raises:
            ValueError: If input_range has zero width.
        """
        input_min, input_max = input_range
        output_min, output_max = output_range

        if input_min == input_max:
            raise ValueError("Input range cannot have zero width")

        # Clip to input range
        clipped = max(input_min, min(input_max, value))

        # Linear scale to output range
        normalized = (clipped - input_min) / (input_max - input_min)
        scaled = output_min + normalized * (output_max - output_min)

        return scaled

    @staticmethod
    def categorical_encode(value: str, mapping: Dict[str, float]) -> float:
        """
        Encode categorical values using a provided mapping.

        Args:
            value: The categorical value to encode.
            mapping: Dictionary mapping category names to float values.

        Returns:
            float: The encoded value.

        Raises:
            KeyError: If value is not in the mapping.
        """
        if value not in mapping:
            raise KeyError(f"'{value}' not found in mapping. Available: {list(mapping.keys())}")
        return mapping[value]


class RollingStats:
    """
    Maintains rolling statistics over a fixed-size window of historical values.

    Stores values in a deque with a maximum length. Computes statistics on demand
    without external dependencies (uses math module only).
    """

    def __init__(self, max_size: int = 252):
        """
        Initialize a RollingStats window.

        Args:
            max_size: Maximum number of values to retain. Default 252 (1 trading year).
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.history: deque = deque(maxlen=max_size)

    def add(self, value: float) -> None:
        """
        Add a new value to the rolling window.

        If the window is full, the oldest value is automatically discarded.

        Args:
            value: The value to add.
        """
        self.history.append(value)

    def mean(self) -> float:
        """
        Compute the mean of all values in the window.

        Returns:
            float: The arithmetic mean.

        Raises:
            ValueError: If no values have been added.
        """
        if not self.history:
            raise ValueError("Cannot compute mean on empty history")
        return sum(self.history) / len(self.history)

    def std(self) -> float:
        """
        Compute the sample standard deviation of all values in the window.

        Uses N-1 (sample std) for population > 1, else returns 0.

        Returns:
            float: The sample standard deviation.

        Raises:
            ValueError: If no values have been added.
        """
        if not self.history:
            raise ValueError("Cannot compute std on empty history")

        if len(self.history) == 1:
            return 0.0

        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self.history) / (len(self.history) - 1)
        return math.sqrt(variance)

    def percentile(self, p: float) -> float:
        """
        Compute the p-th percentile of the window (0 <= p <= 100).

        Uses linear interpolation between sorted values.

        Args:
            p: Percentile value in [0, 100].

        Returns:
            float: The p-th percentile.

        Raises:
            ValueError: If no values or p out of range.
        """
        if not self.history:
            raise ValueError("Cannot compute percentile on empty history")
        if not (0 <= p <= 100):
            raise ValueError("Percentile p must be in [0, 100]")

        sorted_vals = sorted(self.history)
        n = len(sorted_vals)

        # Handle edge cases
        if p == 0:
            return sorted_vals[0]
        if p == 100:
            return sorted_vals[-1]

        # Linear interpolation between closest ranks
        rank = (p / 100.0) * (n - 1)
        lower_idx = int(math.floor(rank))
        upper_idx = int(math.ceil(rank))

        if lower_idx == upper_idx:
            return sorted_vals[lower_idx]

        lower_val = sorted_vals[lower_idx]
        upper_val = sorted_vals[upper_idx]
        frac = rank - lower_idx

        return lower_val + frac * (upper_val - lower_val)

    def min(self) -> float:
        """
        Get the minimum value in the window.

        Returns:
            float: The minimum value.

        Raises:
            ValueError: If no values have been added.
        """
        if not self.history:
            raise ValueError("Cannot compute min on empty history")
        return min(self.history)

    def max(self) -> float:
        """
        Get the maximum value in the window.

        Returns:
            float: The maximum value.

        Raises:
            ValueError: If no values have been added.
        """
        if not self.history:
            raise ValueError("Cannot compute max on empty history")
        return max(self.history)

    def count(self) -> int:
        """
        Get the current number of values in the window.

        Returns:
            int: Number of values currently stored.
        """
        return len(self.history)

    def is_full(self) -> bool:
        """
        Check if the rolling window is at maximum capacity.

        Returns:
            bool: True if count == max_size.
        """
        return len(self.history) == self.max_size

    def reset(self) -> None:
        """Clear all values from the rolling window."""
        self.history.clear()
