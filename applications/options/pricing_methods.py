"""Algorithm-generation dispatch: one call, `price_option(spec, method)`, that
routes a payoff-agnostic `OptionSpec` to whichever pricing generation the
caller asked for (or an honest default), and always returns the same
`PriceResult` shape regardless of which pricer actually ran underneath.

Generations wired in:
  * BLACK_SCHOLES  -- classical.option_pricing.black_scholes (closed form)
  * MONTE_CARLO    -- classical.option_pricing.monte_carlo_pricer /
                       the barrier+basket integrands' own GBM Monte Carlo
  * QAE_MLAE       -- algorithms_extended.qae_options_qiskit_finance
                       (spinqit-adapted MLAE core), via applications.options.
                       option_pricer.EuropeanOptionPricer (call spreads)
  * QAE_IQAE       -- algorithms.qae_iqae's IQAE estimator, via
                       applications.options.option_pricer.DigitalOptionPricer
                       (digital calls) and applications.options.barrier_pricer
                       (barrier/basket, backed by qiskit_finance_extensions)

Not every (payoff, method) pair is meaningful -- there is no vanilla
single-leg European QAE circuit built anywhere in this codebase (only a call
*spread* MLAE circuit and a digital-call IQAE complement exist), and there is
no closed-form Black-Scholes price for a discretely-monitored barrier or a
basket option. Unsupported combinations raise `core.errors.ValidationError`
with an explicit alternative, rather than silently falling back to a
different method than the caller asked for.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from algorithms_extended.qiskit_finance_extensions.resource_estimation import AEResourceEstimate
from applications.options.domain import OptionSpec, PayoffKind
from classical.option_pricing.black_scholes import black_scholes_price
from classical.option_pricing.monte_carlo_pricer import monte_carlo_european_price
from core.errors import ValidationError


class PricingMethod(str, Enum):
    AUTO = "auto"
    BLACK_SCHOLES = "black_scholes"
    MONTE_CARLO = "monte_carlo"
    QAE_MLAE = "qae_mlae"
    QAE_IQAE = "qae_iqae"


# Which methods are structurally supported per payoff kind, for AUTO
# selection and for the "you asked for X but only Y exists" error message.
_SUPPORTED: dict[PayoffKind, tuple[PricingMethod, ...]] = {
    PayoffKind.EUROPEAN_CALL: (PricingMethod.BLACK_SCHOLES, PricingMethod.MONTE_CARLO),
    PayoffKind.EUROPEAN_PUT: (PricingMethod.BLACK_SCHOLES, PricingMethod.MONTE_CARLO),
    PayoffKind.DIGITAL_CALL: (PricingMethod.BLACK_SCHOLES, PricingMethod.MONTE_CARLO, PricingMethod.QAE_IQAE),
    PayoffKind.CALL_SPREAD: (PricingMethod.BLACK_SCHOLES, PricingMethod.MONTE_CARLO, PricingMethod.QAE_MLAE),
    PayoffKind.UP_AND_OUT_CALL: (PricingMethod.MONTE_CARLO, PricingMethod.QAE_IQAE),
    PayoffKind.DOWN_AND_OUT_CALL: (PricingMethod.MONTE_CARLO, PricingMethod.QAE_IQAE),
    PayoffKind.BASKET_CALL: (PricingMethod.MONTE_CARLO, PricingMethod.QAE_IQAE),
}

_AUTO_DEFAULT: dict[PayoffKind, PricingMethod] = {
    PayoffKind.EUROPEAN_CALL: PricingMethod.BLACK_SCHOLES,
    PayoffKind.EUROPEAN_PUT: PricingMethod.BLACK_SCHOLES,
    PayoffKind.DIGITAL_CALL: PricingMethod.QAE_IQAE,
    PayoffKind.CALL_SPREAD: PricingMethod.QAE_MLAE,
    PayoffKind.UP_AND_OUT_CALL: PricingMethod.QAE_IQAE,
    PayoffKind.DOWN_AND_OUT_CALL: PricingMethod.QAE_IQAE,
    PayoffKind.BASKET_CALL: PricingMethod.QAE_IQAE,
}


@dataclass(frozen=True)
class PriceResult:
    """One normalized result shape regardless of which pricer ran."""

    price: float
    method: PricingMethod
    payoff: PayoffKind
    classical_reference_price: float | None
    reference_method: str | None
    abs_error: float | None
    rel_error: float | None
    stderr: float | None
    shots: int | None
    n_qubits: int | None
    wall_time_s: float
    resource_estimate: AEResourceEstimate | None
    note: str
    raw: dict[str, Any] = field(default_factory=dict)


def _check_supported(spec: OptionSpec, method: PricingMethod) -> PricingMethod:
    supported = _SUPPORTED[spec.payoff]
    if method is PricingMethod.AUTO:
        return _AUTO_DEFAULT[spec.payoff]
    if method not in supported:
        raise ValidationError(
            f"{spec.payoff.value} does not support method={method.value}. "
            f"Supported methods for this payoff: {[m.value for m in supported]}. "
            "(There is no vanilla single-leg European QAE circuit in this "
            "codebase -- only a call *spread* MLAE circuit and a digital-call "
            "IQAE complement -- and no closed-form price for a discretely-"
            "monitored barrier or a basket option.)"
        )
    return method


def _price_black_scholes(spec: OptionSpec) -> tuple[float, dict]:
    option_type = "call" if spec.payoff is PayoffKind.EUROPEAN_CALL else "put"
    price = black_scholes_price(s0=spec.spot, k=spec.strike, t=spec.maturity, r=spec.rate,
                                 sigma=spec.vol, option_type=option_type)
    return price, {"option_type": option_type}


def _price_monte_carlo(spec: OptionSpec, *, n_paths: int, seed: int) -> tuple[float, dict]:
    if spec.payoff in (PayoffKind.EUROPEAN_CALL, PayoffKind.EUROPEAN_PUT):
        option_type = "call" if spec.payoff is PayoffKind.EUROPEAN_CALL else "put"
        price, stderr = monte_carlo_european_price(
            s0=spec.spot, k=spec.strike, t=spec.maturity, r=spec.rate, sigma=spec.vol,
            option_type=option_type, n_paths=n_paths, seed=seed,
        )
        return price, {"stderr": stderr, "option_type": option_type}

    if spec.payoff is PayoffKind.CALL_SPREAD:
        from classical.monte_carlo.gbm import gbm_terminal_prices
        import numpy as np

        terminal = gbm_terminal_prices(s0=spec.spot, mu=spec.rate, sigma=spec.vol, t=spec.maturity,
                                        n_paths=n_paths, seed=seed)
        cap = spec.strike2 - spec.strike
        payoff = math.exp(-spec.rate * spec.maturity) * np.clip(terminal - spec.strike, 0, cap)
        price = float(payoff.mean())
        stderr = float(payoff.std(ddof=1) / np.sqrt(n_paths))
        return price, {"stderr": stderr}

    if spec.payoff is PayoffKind.DIGITAL_CALL:
        # No dedicated digital-call Monte Carlo helper exists yet; a
        # cash-or-nothing payoff is a one-line GBM terminal-price indicator,
        # not worth a new classical/ module for -- built inline here.
        from classical.monte_carlo.gbm import gbm_terminal_prices
        import numpy as np

        terminal = gbm_terminal_prices(s0=spec.spot, mu=spec.rate, sigma=spec.vol, t=spec.maturity,
                                        n_paths=n_paths, seed=seed)
        indicator = (terminal > spec.strike).astype(float)
        disc = math.exp(-spec.rate * spec.maturity)
        payoff = disc * spec.payout * indicator
        price = float(payoff.mean())
        stderr = float(payoff.std(ddof=1) / np.sqrt(n_paths))
        return price, {"stderr": stderr}

    if spec.payoff in (PayoffKind.UP_AND_OUT_CALL, PayoffKind.DOWN_AND_OUT_CALL):
        from algorithms_extended.qiskit_finance_extensions.barrier_payoff import build_barrier_payoff_integrand

        integrand = build_barrier_payoff_integrand(
            spec.barrier_kind_str(), s0=spec.spot, strike=spec.strike, barrier=spec.barrier,
            r=spec.rate, sigma=spec.vol, t=spec.maturity, n=4, n_mc_paths=n_paths, seed=seed,
        )
        return integrand.monte_carlo_path_monitored, {
            "stderr": integrand.monte_carlo_path_stderr,
            "terminal_only_parity_price": integrand.monte_carlo_terminal_only,
            "note": "path-monitored (realistic) barrier MC price; terminal_only_parity_price "
                    "in raw[] is the apples-to-apples target the QAE circuit itself prices.",
        }

    if spec.payoff is PayoffKind.BASKET_CALL:
        from algorithms_extended.qiskit_finance_extensions.basket_payoff import build_basket_call_integrand

        integrand = build_basket_call_integrand(
            s0_a=spec.spot, s0_b=spec.spot2, strike=spec.strike, weight_a=spec.weight1,
            weight_b=spec.weight2, r=spec.rate, sigma_a=spec.vol, sigma_b=spec.vol2,
            rho=spec.correlation, t=spec.maturity, n_mc_paths=n_paths, seed=seed,
        )
        return integrand.monte_carlo_price, {"stderr": integrand.monte_carlo_stderr}

    raise ValidationError(f"no Monte Carlo pricer wired for {spec.payoff.value}")


def _price_qae_mlae(spec: OptionSpec, *, shots: int, seed: int, use_noisy: bool) -> tuple[float, dict]:
    from applications.options.option_pricer import EuropeanOptionPricer

    if spec.payoff is not PayoffKind.CALL_SPREAD:
        raise ValidationError("QAE_MLAE only prices call_spread payoffs in this codebase")
    pricer = EuropeanOptionPricer()
    quote = pricer.price_spread(
        S0=spec.spot, K1=spec.strike, K2=spec.strike2, r=spec.rate, sigma=spec.vol, T=spec.maturity,
        shots=shots, seed=seed, use_noisy=use_noisy, skip_noisy=not use_noisy,
    )
    return quote.quantum_price, {"n_grid_points": quote.n_grid_points, "note": quote.note}


def _price_qae_iqae(spec: OptionSpec, *, shots: int, seed: int, n_qubits: int | None,
                     target_epsilon: float) -> tuple[float, dict, AEResourceEstimate | None, int | None]:
    if spec.payoff is PayoffKind.DIGITAL_CALL:
        from applications.options.option_pricer import DigitalOptionPricer

        n = n_qubits or 4
        pricer = DigitalOptionPricer()
        quote = pricer.price_digital_call(
            S0=spec.spot, K=spec.strike, r=spec.rate, sigma=spec.vol, T=spec.maturity, payout=spec.payout,
            n=n, epsilon=target_epsilon, shots=shots,
        )
        return quote.quantum_price_ideal, {
            "discretisation_gap": quote.discretisation_gap, "note": quote.note,
        }, None, n

    if spec.payoff in (PayoffKind.UP_AND_OUT_CALL, PayoffKind.DOWN_AND_OUT_CALL):
        from applications.options.barrier_pricer import BarrierOptionPricer

        n = n_qubits or 4
        pricer = BarrierOptionPricer()
        quote = pricer.price(
            spec.barrier_kind_str(), s0=spec.spot, strike=spec.strike, barrier=spec.barrier,
            r=spec.rate, sigma=spec.vol, t=spec.maturity, n=n, target_epsilon=target_epsilon, seed=seed,
        )
        return quote.quantum_price, {
            "vanilla_upper_bound": quote.vanilla_upper_bound,
            "monte_carlo_terminal_only": quote.monte_carlo_terminal_only,
            "monte_carlo_path_monitored": quote.monte_carlo_path_monitored,
            "note": quote.note,
        }, quote.resource_estimate, n

    if spec.payoff is PayoffKind.BASKET_CALL:
        from applications.options.barrier_pricer import BasketOptionPricer

        n = n_qubits or 2
        pricer = BasketOptionPricer()
        quote = pricer.price(
            s0_a=spec.spot, s0_b=spec.spot2, strike=spec.strike, weight_a=spec.weight1,
            weight_b=spec.weight2, r=spec.rate, sigma_a=spec.vol, sigma_b=spec.vol2,
            rho=spec.correlation, t=spec.maturity, n_a=n, n_b=n, target_epsilon=target_epsilon, seed=seed,
        )
        return quote.quantum_price, {
            "within_mc_ci": quote.within_mc_ci, "monte_carlo_stderr": quote.monte_carlo_stderr,
        }, quote.resource_estimate, n

    raise ValidationError(f"no QAE_IQAE pricer wired for {spec.payoff.value}")


def _classical_reference(spec: OptionSpec, *, n_paths: int, seed: int) -> tuple[float | None, str | None]:
    """Best available classical ground truth for this payoff -- closed form
    where one exists, Monte Carlo otherwise. Used to populate abs_error/
    rel_error on every PriceResult regardless of which method actually ran,
    per this platform's convention of never reporting a quantum price without
    a classical comparator alongside it."""
    if spec.payoff in (PayoffKind.EUROPEAN_CALL, PayoffKind.EUROPEAN_PUT, PayoffKind.DIGITAL_CALL):
        if spec.payoff is PayoffKind.DIGITAL_CALL:
            from algorithms.qae_iqae.finance_integrands import bs_digital_call_price

            return bs_digital_call_price(spec.spot, spec.strike, spec.rate, spec.vol, spec.maturity,
                                          payout=spec.payout), "black_scholes"
        price, _ = _price_black_scholes(spec)
        return price, "black_scholes"
    if spec.payoff is PayoffKind.CALL_SPREAD:
        from algorithms.qae_iqae.finance_integrands import bs_call_price

        price = (bs_call_price(spec.spot, spec.strike, spec.rate, spec.vol, spec.maturity)
                 - bs_call_price(spec.spot, spec.strike2, spec.rate, spec.vol, spec.maturity))
        return price, "black_scholes"
    price, _ = _price_monte_carlo(spec, n_paths=n_paths, seed=seed)
    return price, "monte_carlo_path_monitored" if spec.payoff in (
        PayoffKind.UP_AND_OUT_CALL, PayoffKind.DOWN_AND_OUT_CALL) else "monte_carlo"


def price_option(
    spec: OptionSpec,
    method: PricingMethod = PricingMethod.AUTO,
    *,
    shots: int = 4096,
    seed: int = 42,
    n_qubits: int | None = None,
    target_epsilon: float = 0.02,
    use_noisy: bool = False,
    mc_paths: int = 200_000,
    with_resource_estimate: bool = False,
    log_to_registry: bool = False,
) -> PriceResult:
    """The single entry point every application (scenario grids, resource
    estimation, hedging, the CLI) should call to price an `OptionSpec` --
    never a payoff-specific pricer class directly, so method selection and
    result normalization stay in one place.
    """
    spec = spec.validate_feasible()
    resolved_method = _check_supported(spec, method)

    t0 = time.perf_counter()
    resource_estimate: AEResourceEstimate | None = None
    used_n_qubits: int | None = None
    used_shots: int | None = None

    if resolved_method is PricingMethod.BLACK_SCHOLES:
        if spec.payoff not in (
            PayoffKind.EUROPEAN_CALL, PayoffKind.EUROPEAN_PUT, PayoffKind.DIGITAL_CALL, PayoffKind.CALL_SPREAD,
        ):
            raise ValidationError(
                f"no closed-form Black-Scholes price for {spec.payoff.value} in this codebase "
                "-- use PricingMethod.MONTE_CARLO or PricingMethod.QAE_IQAE"
            )
        if spec.payoff is PayoffKind.DIGITAL_CALL:
            from algorithms.qae_iqae.finance_integrands import bs_digital_call_price

            price = bs_digital_call_price(spec.spot, spec.strike, spec.rate, spec.vol, spec.maturity,
                                           payout=spec.payout)
            raw = {}
        elif spec.payoff is PayoffKind.CALL_SPREAD:
            from algorithms.qae_iqae.finance_integrands import bs_call_price

            price = (bs_call_price(spec.spot, spec.strike, spec.rate, spec.vol, spec.maturity)
                     - bs_call_price(spec.spot, spec.strike2, spec.rate, spec.vol, spec.maturity))
            raw = {}
        else:
            price, raw = _price_black_scholes(spec)
        note = "closed-form Black-Scholes"
    elif resolved_method is PricingMethod.MONTE_CARLO:
        price, raw = _price_monte_carlo(spec, n_paths=mc_paths, seed=seed)
        used_shots = mc_paths
        note = raw.pop("note", "classical GBM Monte Carlo")
    elif resolved_method is PricingMethod.QAE_MLAE:
        price, raw = _price_qae_mlae(spec, shots=shots, seed=seed, use_noisy=use_noisy)
        used_shots = shots
        note = raw.pop("note", "QAE/MLAE (spinqit-adapted core)")
    elif resolved_method is PricingMethod.QAE_IQAE:
        price, raw, resource_estimate_maybe, used_n_qubits = _price_qae_iqae(
            spec, shots=shots, seed=seed, n_qubits=n_qubits, target_epsilon=target_epsilon,
        )
        used_shots = shots
        note = raw.pop("note", "QAE/IQAE (algorithms.qae_iqae + qiskit_finance_extensions)")
        if with_resource_estimate and resource_estimate_maybe is not None:
            resource_estimate = resource_estimate_maybe
    else:  # pragma: no cover - exhaustive Enum
        raise ValidationError(f"unhandled method {resolved_method}")

    wall_time = time.perf_counter() - t0
    ref_price, ref_method = _classical_reference(spec, n_paths=mc_paths, seed=seed)
    abs_err = abs(price - ref_price) if ref_price is not None else None
    rel_err = (abs_err / abs(ref_price)) if (abs_err is not None and ref_price) else None

    result = PriceResult(
        price=price, method=resolved_method, payoff=spec.payoff,
        classical_reference_price=ref_price, reference_method=ref_method,
        abs_error=abs_err, rel_error=rel_err, stderr=raw.pop("stderr", None),
        shots=used_shots, n_qubits=used_n_qubits, wall_time_s=wall_time,
        resource_estimate=resource_estimate, note=note, raw=raw,
    )
    if log_to_registry:
        _log_price_result(spec, result)
    return result


def _log_price_result(spec: OptionSpec, result: PriceResult) -> None:
    """Log a priced request through core.experiment_registry, per this
    build's requirement that every pricing request be traceable the same way
    algorithms.qae_iqae.register.estimate_barrier_probability already logs
    its own runs."""
    from core.experiment_registry import ExperimentRegistry
    from core.schemas import (
        BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata,
    )

    backend = BackendDescriptor(
        name="aer_simulator_noiseless" if result.method != PricingMethod.BLACK_SCHOLES else "classical",
        kind=BackendKind.SIMULATOR, num_qubits=result.n_qubits or 0, provider="local",
    )
    config = ExperimentConfig(
        name=f"applications.options::{spec.payoff.value}::{result.method.value}",
        algorithm=result.method.value, application="options", backend=backend,
        shots=result.shots or 0, seed=42,
        params={"payoff": spec.payoff.value, "spot": spec.spot, "strike": spec.strike},
    )
    metadata = RunMetadata(algorithm=result.method.value, generation="wave5")
    metrics = {"price": result.price}
    if result.abs_error is not None:
        metrics["abs_error"] = result.abs_error
    if result.rel_error is not None:
        metrics["rel_error"] = result.rel_error
    metrics["wall_time_s"] = result.wall_time_s
    classical_metrics = {"reference_price": result.classical_reference_price} if result.classical_reference_price is not None else None
    exp_result = ExperimentResult(
        metadata=metadata, config=config, metrics=metrics,
        classical_baseline_metrics=classical_metrics, notes=result.note,
    )
    ExperimentRegistry().record(exp_result)


__all__ = ["PricingMethod", "PriceResult", "price_option"]
