"""Unified simulator-method selection.

`hardware.backends` already wraps a default `AerSimulator()` (automatic
method) for the noiseless case, and `hardware.noise` builds `NoiseModel`
objects for use with it — but nothing on the platform chose *between*
Aer's simulation methods (statevector, density_matrix, matrix_product_state,
stabilizer) based on circuit/noise characteristics. This module fills that
gap: each method has real, different scaling behavior (statevector is
exponential in qubits regardless of entanglement; density_matrix adds
another exponential factor for exact noise simulation; matrix_product_state
scales with entanglement, not qubit count, for low-entanglement circuits)
and picking the wrong one either wastes time or silently drops noise
fidelity.
"""
from hardware.simulators.method_selection import SimulatorMethod, recommend_method, build_simulator

__all__ = ["SimulatorMethod", "recommend_method", "build_simulator"]
