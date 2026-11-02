"""Real execution paths for the payoff-circuit-based pricers (barrier and
basket options, the only payoffs in this package with an actual amplitude-
estimation *circuit* object to execute -- the call-spread/digital pricers
already run their own circuits internally via `algorithms.qae_iqae.backends`
and `algorithms_extended.qae_options_qiskit_finance`).

Three tiers, one interface:
  * SIMULATOR        -- noiseless Aer, via algorithms_extended.qiskit_finance_
                         extensions.execution.run_payoff_circuit("noiseless")
  * NOISY_SIMULATOR   -- a real Qiskit fake-backend (127+ qubit device
                         calibration snapshot) via the same module's
                         "fake_eagle"/"fake_heron" path -- this is exactly the
                         path that used to OOM before hardware.ibm_runtime.
                         primitives._truncate_to_active_qubits existed (see
                         `verify_qubit_truncation_fix` below and
                         tests/applications/test_options_execution.py).
  * IBM_QPU / BRAKET_* -- credential-gated real hardware. Credentials are
                         checked at the point of submission, never earlier,
                         and a missing credential raises `CredentialError`
                         immediately -- it is never silently swallowed into a
                         dry-run or a fabricated result.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from algorithms.qae_iqae.backends import ancilla_p1
from algorithms_extended.qiskit_finance_extensions.execution import ExecutionResult, run_payoff_circuit
from core.errors import CredentialError, ValidationError


class ExecutionBackend(str, Enum):
    SIMULATOR = "simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    IBM_QPU = "ibm_qpu"
    BRAKET_LOCAL = "braket_local"
    BRAKET_QPU = "braket_qpu"


@dataclass(frozen=True)
class SampledPriceResult:
    """A price recovered from real measurement *counts* (not the statevector-
    exact amplitude `applications.options.barrier_pricer`'s default path
    uses) -- the honest, shot-noise-bearing number an actual circuit
    execution produces."""

    a_hat: float
    price: float
    counts: dict
    shots: int
    backend_name: str
    execution_backend: ExecutionBackend
    mitigation: str


def run_priced_circuit(
    circuit,
    *,
    execution_backend: ExecutionBackend = ExecutionBackend.SIMULATOR,
    shots: int = 4096,
    mitigation: str = "none",
    fake_backend_name: str = "fake_heron",
    ibm_backend_name: str = "ibm_kingston",
    braket_device_arn: str | None = None,
) -> ExecutionResult:
    """Execute a payoff circuit (e.g. from `build_A_barrier_payoff` /
    `build_A_basket_payoff`) on the requested tier and return raw counts.
    Never branches silently past a missing credential -- IBM_QPU and
    BRAKET_QPU raise `CredentialError` up front."""
    if execution_backend is ExecutionBackend.SIMULATOR:
        return run_payoff_circuit(circuit, backend_name="noiseless", shots=shots, mitigation=mitigation)

    if execution_backend is ExecutionBackend.NOISY_SIMULATOR:
        # This is the path that exercises hardware.ibm_runtime.primitives.
        # run_sampler's BackendKind.FAKE branch (via BackendCatalog + the
        # qiskit_finance_extensions.execution wiring), and therefore the
        # _truncate_to_active_qubits qubit-truncation fix -- a real payoff
        # circuit transpiled onto a 127+ qubit fake backend and measured with
        # measure_all() would OOM Aer's dense statevector fallback without it.
        return run_payoff_circuit(circuit, backend_name=fake_backend_name, shots=shots, mitigation=mitigation)

    if execution_backend is ExecutionBackend.IBM_QPU:
        return _run_ibm_qpu(circuit, backend_name=ibm_backend_name, shots=shots)

    if execution_backend is ExecutionBackend.BRAKET_LOCAL:
        return _run_braket_local(circuit, shots=shots)

    if execution_backend is ExecutionBackend.BRAKET_QPU:
        if braket_device_arn is None:
            raise ValidationError("BRAKET_QPU execution requires `braket_device_arn`")
        return _run_braket_remote(circuit, device_arn=braket_device_arn, shots=shots)

    raise ValidationError(f"unhandled execution_backend {execution_backend}")  # pragma: no cover


def _run_ibm_qpu(circuit, *, backend_name: str, shots: int) -> ExecutionResult:
    """Real IBM Quantum Runtime submission. Raises CredentialError before any
    network call if IBM_QUANTUM_TOKEN is absent -- mirrors
    `algorithms.qae_iqae.iqae.run_on_hardware`'s credential gate, but does
    NOT fall back to a token-free dry run: per core.errors.CredentialError's
    contract ("must NEVER be silently swallowed into a no-op fallback"),
    a caller that explicitly asked for IBM_QPU execution and has no
    credential gets a loud failure, not a quietly-substituted simulation."""
    from core.config import get_settings

    settings = get_settings()
    token = settings.ibm.require().token  # raises CredentialError here if absent

    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService(channel=settings.ibm.channel, token=token)
    backend = service.backend(backend_name)
    measured = circuit.copy()
    if measured.num_clbits == 0:
        measured.measure_all()
    transpiled = transpile(measured, backend=backend, seed_transpiler=42)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    result = job.result()
    reg_name = "meas" if hasattr(result[0].data, "meas") else list(result[0].data.__dict__.keys())[0]
    counts = getattr(result[0].data, reg_name).get_counts()
    return ExecutionResult(counts=counts, backend_name=backend.name, mitigation="none", shots=shots)


# hardware.braket.backend_adapter's Qiskit->Braket translator only covers a
# fixed single-/two-qubit gate vocabulary (see its module docstring) -- a
# payoff circuit's `StatePreparation` and multi-controlled-RY instructions
# must be decomposed into that vocabulary before translation, or
# qiskit_to_braket raises UnsupportedGateError on the very first
# unrecognized composite instruction.
_BRAKET_BASIS = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cz", "swap"]


def _decompose_for_braket(circuit):
    from qiskit import transpile

    return transpile(circuit, basis_gates=_BRAKET_BASIS, optimization_level=1)


def _run_braket_local(circuit, *, shots: int) -> ExecutionResult:
    from hardware.braket.backend_adapter import run_local

    measured = _decompose_for_braket(circuit)
    if measured.num_clbits == 0:
        measured.measure_all()
    out = run_local(measured, shots=shots)
    counts = out.get("measurement_counts", {})
    return ExecutionResult(counts=counts, backend_name="braket_local_simulator", mitigation="none", shots=shots)


def _run_braket_remote(circuit, *, device_arn: str, shots: int) -> ExecutionResult:
    """Raises CredentialError before any network call if AWS credentials are
    absent -- see hardware.braket.config.AWSCredentials.require."""
    from hardware.braket.config import get_aws_credentials

    get_aws_credentials().require()  # raises CredentialError here if absent
    from hardware.braket.backend_adapter import run_remote

    measured = _decompose_for_braket(circuit)
    if measured.num_clbits == 0:
        measured.measure_all()
    out = run_remote(measured, device_arn=device_arn, shots=shots)
    counts = out.get("measurement_counts", {})
    return ExecutionResult(counts=counts, backend_name=device_arn, mitigation="none", shots=shots)


def sampled_amplitude(execution_result: ExecutionResult, *, ancilla_clbit: int) -> tuple[float, int]:
    """Extract P(ancilla=1) from measurement counts (works for both raw
    counts and M3-mitigated quasi-probability dicts, since both are
    bitstring -> weight mappings). Returns (a_hat, total_shots_or_weight)."""
    ones, total = ancilla_p1(execution_result.counts, ancilla_clbit=ancilla_clbit)
    if total == 0:
        raise ValidationError("execution_result.counts is empty -- nothing to estimate an amplitude from")
    return ones / total, total


__all__ = [
    "ExecutionBackend", "SampledPriceResult", "run_priced_circuit", "sampled_amplitude",
]
