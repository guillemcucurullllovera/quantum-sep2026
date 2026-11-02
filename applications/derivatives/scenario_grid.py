"""Unified multi-dimensional scenario sweeps across every option payoff kind
this package supports, built on `applications.options.pricing_methods.
price_option` rather than re-deriving pricing logic per payoff.

Distinct from the two existing scenario modules this build found and
deliberately did not duplicate:
  * `applications.derivatives.scenario_analysis.run_spot_vol_sweep`
    -- spot x vol only, call-spread only, classical-grid + selective-quantum.
  * `algorithms_extended.qiskit_finance_extensions.scenario_analysis.
    run_barrier_scenario_sweep` -- spot x vol x barrier, barrier
    payoff only, classical-only (no quantum leg at all).

This module is the payoff-agnostic superset: spot x vol x rate x barrier x
correlation, any payoff kind, with an explicit, cost-aware knob for how many
of the grid cells actually get a (slow) quantum re-price versus how many stay
classical-only -- the same "quantum coverage is a stated, deliberate subset"
convention `scenario_analysis.py`'s docstring already establishes for this
codebase, generalized across dimensions and payoffs instead of re-invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

from applications.options.domain import OptionSpec, PayoffKind
from applications.options.pricing_methods import PriceResult, PricingMethod, price_option
from core.errors import ValidationError


@dataclass(frozen=True)
class ScenarioCell:
    overrides: dict[str, float]
    classical: PriceResult
    quantum: PriceResult | None = None


@dataclass(frozen=True)
class ScenarioGridResult:
    base_spec: OptionSpec
    dimensions: dict[str, tuple[float, ...]]
    cells: list[ScenarioCell] = field(default_factory=list)

    def as_table(self) -> list[dict]:
        rows = []
        for cell in self.cells:
            row = dict(cell.overrides)
            row["classical_price"] = cell.classical.price
            if cell.quantum is not None:
                row["quantum_price"] = cell.quantum.price
                row["quantum_vs_classical_abs_error"] = abs(cell.quantum.price - cell.classical.price)
            rows.append(row)
        return rows


_ATTR_TO_SPEC_FIELD = {
    "spot": "spot", "vol": "vol", "rate": "rate", "barrier": "barrier", "correlation": "correlation",
}


def run_scenario_grid(
    base_spec: OptionSpec,
    *,
    spots: Sequence[float] | None = None,
    vols: Sequence[float] | None = None,
    rates: Sequence[float] | None = None,
    barriers: Sequence[float] | None = None,
    correlations: Sequence[float] | None = None,
    classical_method: PricingMethod = PricingMethod.MONTE_CARLO,
    quantum_method: PricingMethod | None = None,
    quantum_cell_indices: Sequence[int] = (),
    mc_paths: int = 20_000,
    shots: int = 2048,
    n_qubits: int | None = None,
    seed: int = 42,
) -> ScenarioGridResult:
    """Sweep any combination of spot/vol/rate/barrier/correlation, pricing
    every grid cell classically and, for the caller-selected subset of cells
    (`quantum_cell_indices`, 0-indexed into the flattened grid, row-major
    over the dimensions passed), also with a quantum method.

    `barriers`/`correlations` are only meaningful (and only accepted) for
    payoffs that have that field -- passing `barriers` for a `european_call`
    `base_spec` raises `ValidationError` rather than silently ignoring it.
    """
    base_spec = base_spec.validate_feasible()
    dims: dict[str, tuple[float, ...]] = {}
    if spots is not None:
        dims["spot"] = tuple(spots)
    if vols is not None:
        dims["vol"] = tuple(vols)
    if rates is not None:
        dims["rate"] = tuple(rates)
    if barriers is not None:
        if base_spec.payoff not in (PayoffKind.UP_AND_OUT_CALL, PayoffKind.DOWN_AND_OUT_CALL):
            raise ValidationError(f"barriers sweep is only meaningful for barrier payoffs, not {base_spec.payoff.value}")
        dims["barrier"] = tuple(barriers)
    if correlations is not None:
        if base_spec.payoff is not PayoffKind.BASKET_CALL:
            raise ValidationError(f"correlations sweep is only meaningful for basket_call, not {base_spec.payoff.value}")
        dims["correlation"] = tuple(correlations)
    if not dims:
        raise ValidationError("run_scenario_grid needs at least one of spots/vols/rates/barriers/correlations")

    dim_names = list(dims.keys())
    combos = list(product(*[dims[name] for name in dim_names]))
    quantum_idx = set(quantum_cell_indices)

    cells: list[ScenarioCell] = []
    for i, combo in enumerate(combos):
        overrides = dict(zip(dim_names, combo))
        spec_i = base_spec.model_copy(update={_ATTR_TO_SPEC_FIELD[k]: v for k, v in overrides.items()})
        spec_i = spec_i.validate_feasible()

        classical = price_option(spec_i, classical_method, mc_paths=mc_paths, seed=seed)
        quantum = None
        if i in quantum_idx:
            qm = quantum_method or PricingMethod.AUTO
            quantum = price_option(spec_i, qm, shots=shots, seed=seed, n_qubits=n_qubits)
        cells.append(ScenarioCell(overrides=overrides, classical=classical, quantum=quantum))

    return ScenarioGridResult(base_spec=base_spec, dimensions=dims, cells=cells)


__all__ = ["ScenarioGridResult", "ScenarioCell", "run_scenario_grid"]
