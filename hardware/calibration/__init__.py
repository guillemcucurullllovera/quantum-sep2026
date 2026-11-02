"""Calibration-data ingestion and historical tracking.

Distinct from `hardware.qubit_selection`, which *consumes* already-parsed
calibration numbers to score qubits — this module is where those numbers
first get extracted out of a raw Qiskit `Target` and, optionally, tracked
over time so calibration drift becomes observable.
"""
from hardware.calibration.ingest import CalibrationSnapshot, extract_calibration
from hardware.calibration.history import CalibrationHistory

__all__ = ["CalibrationSnapshot", "extract_calibration", "CalibrationHistory"]
