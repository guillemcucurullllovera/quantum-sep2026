"""Application-layer option pricers. Every price returned here is paired with its
classical Black-Scholes ground truth in the same result object -- this platform never
reports a quantum price without its classical comparator alongside it.
"""
from __future__ import annotations

from dataclasses import dataclass

from algorithms.qae_iqae.finance_integrands import bs_call_price, build_call_spread_integrand
from algorithms_extended.qae_options_qiskit_finance.option_pricing import (
    CallSpreadPricingResult,
    DigitalCallPricingResult,
    digital_call_probability,
    run_call_spread_mlae,
)
from core.registry import APPLICATION_REGISTRY


@dataclass(frozen=True)
class EuropeanOptionQuote:
    quantum_price: float
    classical_price: float
    abs_error: float
    rel_error: float
    n_grid_points: int
    note: str


class EuropeanOptionPricer:
    """Prices a European bull call spread via QAE/MLAE, benchmarked against the
    closed-form Black-Scholes spread price on every call."""

    def price_spread(
        self, S0: float = 100.0, K1: float = 95.0, K2: float = 115.0,
        r: float = 0.04, sigma: float = 0.20, T: float = 0.25,
        *, shots: int = 4096, seed: int = 42, use_noisy: bool = False, n_bootstrap: int = 500,
        skip_noisy: bool = False,
    ) -> EuropeanOptionQuote:
        integrand = build_call_spread_integrand(S0=S0, K1=K1, K2=K2, r=r, sigma=sigma, T=T)
        result: CallSpreadPricingResult = run_call_spread_mlae(
            integrand, shots=shots, seed=seed, n_bootstrap=n_bootstrap,
            skip_noisy=skip_noisy or not use_noisy,
        )
        quantum_price = result.quantum_price_noisy if use_noisy else result.quantum_price_ideal
        classical = result.true_bs_spread_price
        abs_err = abs(quantum_price - classical)
        return EuropeanOptionQuote(
            quantum_price=quantum_price, classical_price=classical,
            abs_error=abs_err, rel_error=abs_err / classical if classical else float("nan"),
            n_grid_points=result.n_grid_points, note=result.note,
        )


class DigitalOptionPricer:
    """Prices a cash-or-nothing digital call via the IQAE barrier estimator's
    complement, benchmarked against the closed-form Black-Scholes digital price."""

    def price_digital_call(
        self, S0: float = 100.0, K: float = 100.0, r: float = 0.04,
        sigma: float = 0.20, T: float = 0.25, payout: float = 1.0,
        *, n: int = 3, epsilon: float = 0.01, shots: int = 2048,
    ) -> DigitalCallPricingResult:
        return digital_call_probability(S0, K, r, sigma, T, payout, n=n, epsilon=epsilon, shots=shots)


def _register() -> None:
    if "options" not in APPLICATION_REGISTRY.list():
        APPLICATION_REGISTRY.register("options")(EuropeanOptionPricer)


_register()
