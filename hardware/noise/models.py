"""Noise model construction: real device-calibrated noise (from a fake
backend's snapshot) and a controllable synthetic noise model for isolated
experiments where you want to sweep a single error parameter.
"""
from __future__ import annotations

from core.schemas import NoiseModelDescriptor


def from_fake_backend(fake_backend) -> tuple[object, NoiseModelDescriptor]:
    """Build an Aer NoiseModel from a fake backend's calibration snapshot,
    plus a typed descriptor summarizing it (mean readout/gate error) so
    experiment results can report what noise conditions actually applied.
    """
    from qiskit_aer.noise import NoiseModel

    noise_model = NoiseModel.from_backend(fake_backend)
    readout_errors = []
    gate_errors = []
    target = fake_backend.target
    for q in range(fake_backend.num_qubits):
        try:
            readout_errors.append(float(target["measure"][(q,)].error))
        except (KeyError, AttributeError, TypeError):
            continue
    for gate_name in target.operation_names:
        for qargs, props in (target[gate_name] or {}).items():
            if props is not None and props.error is not None:
                gate_errors.append(float(props.error))

    descriptor = NoiseModelDescriptor(
        backend_name=getattr(fake_backend, "name", str(fake_backend)),
        readout_error_mean=(sum(readout_errors) / len(readout_errors)) if readout_errors else None,
        gate_error_mean=(sum(gate_errors) / len(gate_errors)) if gate_errors else None,
        source="fake_backend",
    )
    return noise_model, descriptor


def synthetic(
    depolarizing_prob_1q: float = 0.001,
    depolarizing_prob_2q: float = 0.01,
    readout_error_prob: float = 0.02,
) -> tuple[object, NoiseModelDescriptor]:
    """A controllable synthetic noise model (uniform depolarizing +
    symmetric readout error) for experiments that need to sweep a single
    noise parameter in isolation rather than inheriting a fixed device
    calibration snapshot.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

    noise_model = NoiseModel()
    err_1q = depolarizing_error(depolarizing_prob_1q, 1)
    err_2q = depolarizing_error(depolarizing_prob_2q, 2)
    noise_model.add_all_qubit_quantum_error(err_1q, ["sx", "x", "rz", "id"])
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "ecr"])
    ro_error = ReadoutError(
        [[1 - readout_error_prob, readout_error_prob], [readout_error_prob, 1 - readout_error_prob]]
    )
    noise_model.add_all_qubit_readout_error(ro_error)

    descriptor = NoiseModelDescriptor(
        backend_name="synthetic",
        readout_error_mean=readout_error_prob,
        gate_error_mean=(depolarizing_prob_1q + depolarizing_prob_2q) / 2,
        source="synthetic",
    )
    return noise_model, descriptor


__all__ = ["from_fake_backend", "synthetic"]
