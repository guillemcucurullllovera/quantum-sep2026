"""
quantum_signal: Core quantum-inspired signal processing for multi-modal market data.

This package provides tools for ingesting, validating, and transforming
market signals from multiple agents into quantum-ready vector representations.
"""

from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector
from prototypes.other.quantum_signal_gen1_infra.logger import build_quantum_vector

__version__ = "0.1.0"
__all__ = [
    "QuantumInputVector",
    "build_quantum_vector",
]
