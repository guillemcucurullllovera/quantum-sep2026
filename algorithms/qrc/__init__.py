"""Quantum Reservoir Computing — canonical (Gen2-informed) implementation.

See algorithms_extended/qrc_arxiv_replication/ for the paper-faithful port
kept separately for replication validity, and that module's THIRD_PARTY_NOTICES.md
for the honest validation history of the QRC approach generally.
"""
from algorithms.qrc.reservoir import QRCAlgorithm, QuantumReservoir, QuantumReservoirConfig

__all__ = ["QuantumReservoir", "QuantumReservoirConfig", "QRCAlgorithm"]
