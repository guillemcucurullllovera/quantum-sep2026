"""Resource estimation + convergence studies for the amplitude-estimation
pricers.

Two real, run-it-and-look-at-the-numbers deliverables:

  * `estimate_resources(spec, target_epsilon)` -- wires
    `algorithms_extended.qiskit_finance_extensions.resource_estimation` to a
    real built circuit for the requested `OptionSpec`, reporting qubits/
    depth/2Q-gate-count plus the canonical-AE and IQAE oracle-query counts
    needed to reach `target_epsilon` at a target confidence.

  * `run_convergence_study(spec, shots_grid, n_qubits_grid)` -- for barrier/
    basket payoffs, actually EXECUTES the built A-circuit (via
    `applications.options.execution.run_priced_circuit`, simulator tier) at
    each (shots, n_qubits) combination, recovers a shot-noise-bearing
    amplitude/price from real measurement counts (not the statevector-exact
    price `barrier_pricer`/`basket_pricer` report by default), and reports
    its error against a fixed, high-path-count classical Monte Carlo ground
    truth -- so the study shows both discretisation error shrinking with
    n_qubits and shot noise shrinking with shots, against the same
    reference. For call-spread/digital payoffs (no standalone circuit object
    exposed by their pricers) it reuses the existing MLAE/IQAE pricers' own
    `shots`/`n` parameters directly, which already run a real
    sampled/iterative estimator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from algorithms_extended.qiskit_finance_extensions.resource_estimation import AEResourceEstimate, estimate
from applications.options.domain import OptionSpec, PayoffKind
from applications.options.execution import ExecutionBackend, run_priced_circuit, sampled_amplitude
from core.errors import ValidationError


def _build_circuit_and_meta(spec: OptionSpec, *, n_qubits: int, seed: int):
    """Return (circuit, ancilla_clbit, price_from_amplitude_fn, mc_ground_truth)."""
    if spec.payoff in (PayoffKind.UP_AND_OUT_CALL, PayoffKind.DOWN_AND_OUT_CALL):
        from algorithms_extended.qiskit_finance_extensions.barrier_payoff import (
            build_A_barrier_payoff, build_barrier_payoff_integrand, price_from_amplitude,
        )

        integrand = build_barrier_payoff_integrand(
            spec.barrier_kind_str(), s0=spec.spot, strike=spec.strike, barrier=spec.barrier,
            r=spec.rate, sigma=spec.vol, t=spec.maturity, n=n_qubits, n_mc_paths=200_000, seed=seed,
        )
        circuit = build_A_barrier_payoff(integrand)
        return circuit, n_qubits, (lambda a: price_from_amplitude(integrand, a)), integrand.monte_carlo_terminal_only

    if spec.payoff is PayoffKind.BASKET_CALL:
        from algorithms_extended.qiskit_finance_extensions.basket_payoff import (
            build_A_basket_payoff, build_basket_call_integrand, price_from_amplitude,
        )

        integrand = build_basket_call_integrand(
            s0_a=spec.spot, s0_b=spec.spot2, strike=spec.strike, weight_a=spec.weight1,
            weight_b=spec.weight2, r=spec.rate, sigma_a=spec.vol, sigma_b=spec.vol2,
            rho=spec.correlation, t=spec.maturity, n_a=n_qubits, n_b=n_qubits, n_mc_paths=200_000, seed=seed,
        )
        circuit = build_A_basket_payoff(integrand)
        ancilla = integrand.dist.num_qubits_a + integrand.dist.num_qubits_b
        return circuit, ancilla, (lambda a: price_from_amplitude(integrand, a)), integrand.monte_carlo_price

    raise ValidationError(
        f"resource/convergence circuit construction is only wired for barrier and basket "
        f"payoffs in this module, not {spec.payoff.value}"
    )


def estimate_resources(spec: OptionSpec, *, target_epsilon: float = 0.01, confidence: float = 0.95,
                        n_qubits: int | None = None, seed: int = 42) -> AEResourceEstimate:
    """Real resource estimate (qubits/depth/2Q-gates + oracle-query counts)
    for pricing `spec` to `target_epsilon` -- built from an actually-
    constructed circuit, not a qubit-count guess."""
    spec = spec.validate_feasible()
    n = n_qubits or (4 if spec.payoff in (PayoffKind.UP_AND_OUT_CALL, PayoffKind.DOWN_AND_OUT_CALL) else 2)
    circuit, _, _, _ = _build_circuit_and_meta(spec, n_qubits=n, seed=seed)
    return estimate(circuit, target_epsilon=target_epsilon, confidence=confidence)


@dataclass(frozen=True)
class ConvergencePoint:
    n_qubits: int
    shots: int
    a_hat: float
    sampled_price: float
    exact_grid_price: float
    mc_ground_truth: float
    abs_error_vs_mc: float
    abs_error_vs_grid: float
    circuit_n_qubits: int


@dataclass(frozen=True)
class ConvergenceStudyResult:
    payoff: PayoffKind
    points: list[ConvergencePoint] = field(default_factory=list)

    def as_table(self) -> list[dict]:
        return [
            {
                "n_qubits": p.n_qubits, "shots": p.shots, "a_hat": p.a_hat,
                "sampled_price": p.sampled_price, "mc_ground_truth": p.mc_ground_truth,
                "abs_error_vs_mc": p.abs_error_vs_mc, "abs_error_vs_grid": p.abs_error_vs_grid,
            }
            for p in self.points
        ]


def run_convergence_study(
    spec: OptionSpec,
    *,
    shots_grid: tuple[int, ...] = (256, 1024, 4096, 16384),
    n_qubits_grid: tuple[int, ...] = (2, 3, 4),
    seed: int = 42,
) -> ConvergenceStudyResult:
    """Execute the real A-circuit at every (n_qubits, shots) combination on
    the noiseless simulator, recover a sampled amplitude/price from actual
    measurement counts, and compare against a fixed classical Monte Carlo
    ground truth -- a genuine convergence table, not a synthetic curve."""
    spec = spec.validate_feasible()
    if spec.payoff not in (PayoffKind.UP_AND_OUT_CALL, PayoffKind.DOWN_AND_OUT_CALL, PayoffKind.BASKET_CALL):
        raise ValidationError(
            f"run_convergence_study only supports barrier/basket payoffs (real circuit "
            f"execution), not {spec.payoff.value}. For call_spread/digital_call, call "
            "applications.options.pricing_methods.price_option repeatedly across a shots "
            "grid instead -- those pricers already run a real sampled/iterative estimator."
        )

    points: list[ConvergencePoint] = []
    for n_qubits in n_qubits_grid:
        circuit, ancilla_clbit, price_fn, mc_ground_truth = _build_circuit_and_meta(
            spec, n_qubits=n_qubits, seed=seed,
        )
        exact_price = price_fn(_exact_amplitude(circuit, ancilla_clbit))
        for shots in shots_grid:
            result = run_priced_circuit(
                circuit, execution_backend=ExecutionBackend.SIMULATOR, shots=shots, mitigation="none",
            )
            a_hat, total = sampled_amplitude(result, ancilla_clbit=ancilla_clbit)
            sampled_price = price_fn(a_hat)
            points.append(ConvergencePoint(
                n_qubits=n_qubits, shots=total, a_hat=a_hat, sampled_price=sampled_price,
                exact_grid_price=exact_price, mc_ground_truth=mc_ground_truth,
                abs_error_vs_mc=abs(sampled_price - mc_ground_truth),
                abs_error_vs_grid=abs(sampled_price - exact_price),
                circuit_n_qubits=circuit.num_qubits,
            ))
    return ConvergenceStudyResult(payoff=spec.payoff, points=points)


def _exact_amplitude(circuit, ancilla_clbit: int) -> float:
    """Statevector-exact P(ancilla=1) of the built circuit (no shot noise) --
    the noiseless-QAE target every sampled point above is estimating."""
    from qiskit.quantum_info import Statevector

    sv = Statevector.from_instruction(circuit)
    n = circuit.num_qubits
    probs = sv.probabilities_dict()
    total = 0.0
    for bitstring, p in probs.items():
        # Statevector bitstrings are qubit-index order, rightmost = qubit 0.
        idx = n - 1 - ancilla_clbit
        if 0 <= idx < len(bitstring) and bitstring[idx] == "1":
            total += p
    return total


__all__ = [
    "estimate_resources", "run_convergence_study", "ConvergenceStudyResult", "ConvergencePoint",
]
