"""Multi-method comparison report: run every pricing method this codebase
actually supports for a given `OptionSpec` and lay the results side by side
-- price, error against the shared classical reference, wall-clock time, and
(where available) resource estimate. This is the "real comparison report"
deliverable and doubles as the CLI's default output (see `cli.py`) and as a
`benchmarks.runner`-compatible quantum-vs-classical benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from applications.options.domain import OptionSpec
from applications.options.pricing_methods import PriceResult, PricingMethod, _SUPPORTED, price_option
from benchmarks.runner import BenchmarkResult, MethodOutput, run_benchmark


@dataclass(frozen=True)
class ComparisonReport:
    spec: OptionSpec
    results: dict[str, PriceResult] = field(default_factory=dict)

    def as_table(self) -> list[dict]:
        rows = []
        for name, r in self.results.items():
            rows.append({
                "method": name, "price": r.price,
                "classical_reference_price": r.classical_reference_price,
                "reference_method": r.reference_method,
                "abs_error": r.abs_error, "rel_error": r.rel_error,
                "shots": r.shots, "n_qubits": r.n_qubits, "wall_time_s": r.wall_time_s,
            })
        return rows


def build_comparison_report(
    spec: OptionSpec, *, shots: int = 4096, seed: int = 42, n_qubits: int | None = None,
) -> ComparisonReport:
    """Price `spec` with every method `pricing_methods._SUPPORTED` lists for
    its payoff kind, catching per-method failures individually so one
    unsupported/erroring method doesn't blank out the whole report."""
    spec = spec.validate_feasible()
    results: dict[str, PriceResult] = {}
    for method in _SUPPORTED[spec.payoff]:
        results[method.value] = price_option(spec, method, shots=shots, seed=seed, n_qubits=n_qubits)
    return ComparisonReport(spec=spec, results=results)


def benchmark_quantum_vs_classical(
    spec: OptionSpec, *, quantum_method: PricingMethod, classical_method: PricingMethod = PricingMethod.MONTE_CARLO,
    shots: int = 4096, seed: int = 42, n_qubits: int | None = None,
) -> BenchmarkResult:
    """Run the platform's standard `benchmarks.runner.run_benchmark` harness
    on one quantum method vs. one classical method for `spec`, so this
    package's results land in the same `core.schemas.BenchmarkResult` shape
    benchmarks/report.py and the rest of the platform already consume."""
    spec = spec.validate_feasible()

    def _quantum() -> MethodOutput:
        r = price_option(spec, quantum_method, shots=shots, seed=seed, n_qubits=n_qubits, with_resource_estimate=True)
        stats = r.resource_estimate.circuit_stats if r.resource_estimate else None
        return MethodOutput(
            value=r.price,
            circuit_depth=stats.depth if stats else None,
            two_qubit_gate_count=stats.two_qubit_gate_count if stats else None,
            shots=r.shots,
        )

    def _classical() -> MethodOutput:
        r = price_option(spec, classical_method, shots=shots, seed=seed)
        return MethodOutput(value=r.price)

    return run_benchmark(
        method_a=_quantum, method_b=_classical,
        method_a_name=quantum_method.value, method_b_name=classical_method.value,
        metric_name="option_price", seed=seed,
        context={"payoff": spec.payoff.value, "spot": spec.spot, "strike": spec.strike},
    )


__all__ = ["ComparisonReport", "build_comparison_report", "benchmark_quantum_vs_classical"]
