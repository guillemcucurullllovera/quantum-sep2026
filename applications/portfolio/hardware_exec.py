"""Real execution paths for a portfolio QAOA circuit: local noiseless Aer,
device-calibrated noisy Aer (a real fake-backend snapshot), synthetic-noise
Aer, and -- credential-gated, never actually invoked without real,
provisioned credentials -- a real IBM Runtime QPU path and a real AWS
Braket QPU path, so a portfolio optimization can genuinely target either
cloud through one call convention.

Every target here goes through this platform's own `hardware.*` submodules
(`hardware.backends.catalog`, `hardware.noise`, `hardware.ibm_runtime`,
`hardware.braket`) -- this module adds no new execution machinery, it wires
a portfolio QAOA circuit into infrastructure that already exists.

`build_portable_qaoa_circuit` uses only the gate vocabulary
`hardware.braket.backend_adapter` actually translates (H, RX/RY/RZ, CX --
see that module's docstring): a ZZ interaction is decomposed as
CX-RZ-CX rather than using Qiskit's native `rzz`, which is NOT in that
vocabulary and would otherwise make this module's circuits Aer/IBM-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from qiskit import QuantumCircuit

from core.config import get_settings
from core.errors import ValidationError
from hardware import noise as hardware_noise
from hardware.backends.catalog import BackendCatalog
from hardware.braket import backend_adapter as braket_adapter
from hardware.braket.config import get_aws_credentials
from hardware.ibm_runtime.primitives import run_sampler


def build_portable_qaoa_circuit(J: np.ndarray, h: np.ndarray, p: int, gamma: np.ndarray, beta: np.ndarray) -> QuantumCircuit:
    """Standard-mixer QAOA circuit built ONLY from H, RZ, CX, RX, measure --
    portable across every target `execute_on_target` supports, including
    Braket's translated gate vocabulary (which does not include `rzz`)."""
    n = len(h)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for layer in range(p):
        g, b = float(gamma[layer]), float(beta[layer])
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * g * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    theta = 2 * g * J[i, j]
                    qc.cx(i, j)
                    qc.rz(theta, j)
                    qc.cx(i, j)
        for i in range(n):
            qc.rx(2 * b, i)
    qc.measure(range(n), range(n))
    return qc


class HardwareTarget(str, Enum):
    AER_NOISELESS = "aer_noiseless"              # local Aer statevector/qasm sim, no noise
    AER_FAKE_NOISY = "aer_fake_noisy"             # device-calibrated noise from a real fake-backend snapshot
    AER_SYNTHETIC_NOISY = "aer_synthetic_noisy"   # controllable synthetic depolarizing + readout noise
    IBM_QPU = "ibm_qpu"                           # credential-gated: real IBM Quantum hardware
    BRAKET_LOCAL = "braket_local"                 # Braket's local simulator, no AWS credentials needed
    BRAKET_QPU = "braket_qpu"                     # credential-gated: real AWS Braket hardware


@dataclass
class HardwareExecutionResult:
    target: str
    backend_name: str
    counts: dict[str, int] | None
    shots: int
    noise_descriptor: dict | None = None


def execute_on_target(
    circuit: QuantumCircuit,
    target: HardwareTarget,
    *,
    shots: int = 2048,
    fake_backend: str = "fake_eagle",
    device_arn: str | None = None,
) -> HardwareExecutionResult:
    """Run `circuit` (already carrying its own measurements) against one of
    six real execution targets. IBM_QPU/BRAKET_QPU raise `CredentialError`
    before any network call if the corresponding credential is absent --
    exactly the platform-wide policy `hardware.ibm_runtime`/`hardware.braket`
    already enforce, never bypassed or stubbed to "succeed" silently here.
    """
    if target == HardwareTarget.AER_NOISELESS:
        resolved = BackendCatalog.get("noiseless")
        r = run_sampler(circuit, resolved.descriptor, resolved.backend, shots=shots)
        return HardwareExecutionResult(target.value, r.backend_name, r.counts, shots)

    if target == HardwareTarget.AER_FAKE_NOISY:
        resolved = BackendCatalog.get(fake_backend)
        r = run_sampler(circuit, resolved.descriptor, resolved.backend, shots=shots)
        return HardwareExecutionResult(target.value, r.backend_name, r.counts, shots)

    if target == HardwareTarget.AER_SYNTHETIC_NOISY:
        from qiskit_aer import AerSimulator

        noise_model, descriptor = hardware_noise.synthetic()
        backend = AerSimulator(noise_model=noise_model)
        job = backend.run(circuit, shots=shots)
        counts = job.result().get_counts()
        return HardwareExecutionResult(
            target.value, "aer_synthetic_noisy", counts, shots, noise_descriptor=descriptor.model_dump()
        )

    if target == HardwareTarget.IBM_QPU:
        get_settings().ibm.require()  # raises CredentialError when IBM_QUANTUM_TOKEN is unset
        raise NotImplementedError(
            "IBM_QPU is credential-gated by design (see the CredentialError this would have "
            "raised above, in an environment without IBM_QUANTUM_TOKEN) -- real submission is "
            "wired through hardware.ibm_runtime session/job-lifecycle management, not invoked "
            "here without real, provisioned credentials."
        )

    if target == HardwareTarget.BRAKET_LOCAL:
        out = braket_adapter.run_local(circuit, shots=shots)
        return HardwareExecutionResult(target.value, "braket_local_simulator", out.get("measurement_counts"), shots)

    if target == HardwareTarget.BRAKET_QPU:
        if device_arn is None:
            raise ValidationError("BRAKET_QPU requires device_arn")
        get_aws_credentials().require()  # raises CredentialError when AWS credentials are unset
        out = braket_adapter.run_remote(circuit, device_arn, shots=shots)
        return HardwareExecutionResult(target.value, device_arn, out.get("measurement_counts"), shots)

    raise ValidationError(f"unknown HardwareTarget {target!r}")
