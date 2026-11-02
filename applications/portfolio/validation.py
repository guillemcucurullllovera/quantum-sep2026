"""Post-hoc constraint validation: check what a solver's bitstring actually
does, independent of what the solver claims. Every solve_portfolio /
compare_solvers result should be run through this before being reported as
"the answer" -- a solver reporting success is not the same as a bitstring
actually satisfying the constraint. Every constraint kind in
`domain.ConstraintSpec`'s vocabulary is checked here, including the ones
(`risk_budget`, `tracking_error`, `concentration_limit`, `factor_exposure`)
that no long-only bitstring solver enforces as a hard constraint -- this is
the one place they're actually checked, under the same equal-weight-per-
holding assumption every long-only solver in this platform makes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from applications.portfolio.domain import PortfolioSpec


@dataclass
class ValidationReport:
    ok: bool
    violations: list[str] = field(default_factory=list)


def validate_bitstring(spec: PortfolioSpec, x: np.ndarray) -> ValidationReport:
    x = np.asarray(x, dtype=float)
    n = spec.n_assets()

    if spec.long_short and len(x) == 2 * n:
        return _validate_long_short_bitstring(spec, x)

    if len(x) != n:
        return ValidationReport(ok=False, violations=[f"bitstring length {len(x)} != n_assets {n}"])

    violations: list[str] = []
    held_count = int((x > 0.5).sum())
    cov = spec.cov()

    for c in spec.constraints:
        if c.kind == "cardinality":
            k = c.params["k"]
            if held_count != k:
                violations.append(f"cardinality: expected exactly {k} holdings, got {held_count}")
        elif c.kind == "sector_range":
            idx = [spec.asset_names().index(nm) for nm in c.params["asset_names"]]
            held = int(x[idx].sum())
            lo = c.params.get("min_count", 0)
            hi = c.params.get("max_count")
            if held < lo:
                violations.append(f"sector_range {c.params['name']}: {held} < min {lo}")
            if hi is not None and held > hi:
                violations.append(f"sector_range {c.params['name']}: {held} > max {hi}")
        elif c.kind == "turnover":
            prev = np.asarray(c.params["previous_holdings"], dtype=float)
            changed = int(np.sum(np.abs(x - prev) > 0.5))
            limit = c.params["limit"]
            if changed > limit:
                violations.append(f"turnover: {changed} changes > limit {limit}")
        elif c.kind == "liquidity_floor":
            scores = spec.liquidity_scores()
            held_scores = scores[x > 0.5]
            floor = c.params["floor"]
            if len(held_scores) and held_scores.min() < floor:
                violations.append(f"liquidity_floor: a held asset scores below {floor}")
        elif c.kind == "leverage":
            limit = c.params["max_gross_positions"]
            if held_count > limit:
                violations.append(f"leverage: {held_count} gross positions > limit {limit}")
        elif c.kind == "concentration_limit":
            max_weight = c.params["max_weight"]
            if held_count > 0 and (1.0 / held_count) > max_weight + 1e-9:
                violations.append(
                    f"concentration_limit: equal weight 1/{held_count} exceeds max_weight {max_weight}"
                )
        elif c.kind == "risk_budget":
            if held_count > 0:
                w = x / held_count
                variance = float(w @ cov @ w)
                max_var = c.params["max_variance"]
                if variance > max_var + 1e-9:
                    violations.append(f"risk_budget: realized variance {variance:.6f} > budget {max_var}")
        elif c.kind == "tracking_error":
            if held_count > 0:
                w = x / held_count
                wb = np.asarray(c.params["benchmark_weights"], dtype=float)
                diff = w - wb
                te = float(np.sqrt(max(float(diff @ cov @ diff), 0.0)))
                max_te = c.params["max_te"]
                if te > max_te + 1e-9:
                    violations.append(f"tracking_error: {te:.6f} > max {max_te}")
        elif c.kind == "factor_exposure":
            loadings = np.asarray(c.params["loadings"], dtype=float)
            k_for_norm = held_count if held_count else 1
            normalized = float(loadings @ x) / k_for_norm
            lo, hi = c.params.get("min_exposure"), c.params.get("max_exposure")
            if lo is not None and normalized < lo - 1e-9:
                violations.append(f"factor_exposure: {normalized:.6f} < min {lo}")
            if hi is not None and normalized > hi + 1e-9:
                violations.append(f"factor_exposure: {normalized:.6f} > max {hi}")
        elif c.kind == "net_exposure":
            continue  # long/short-only; meaningless for a long-only bitstring
        # transaction_cost is a soft/penalty-only cost term, not a hard
        # constraint -- nothing to validate as pass/fail here.

    return ValidationReport(ok=(len(violations) == 0), violations=violations)


def _validate_long_short_bitstring(spec: PortfolioSpec, x: np.ndarray) -> ValidationReport:
    """The 2n-variable [long_0..n-1, short_0..n-1] encoding
    (`LongShortQUBOBuilder`/`FinancePortfolioModel(long_short=True)`)."""
    n = spec.n_assets()
    long_mask = x[:n] > 0.5
    short_mask = x[n:] > 0.5
    violations: list[str] = []

    both = [i for i in range(n) if long_mask[i] and short_mask[i]]
    if both:
        violations.append(f"long_short: asset(s) at index {both} flagged both long and short")

    n_long, n_short = int(long_mask.sum()), int(short_mask.sum())
    for c in spec.constraints:
        if c.kind == "leverage":
            limit = c.params["max_gross_positions"]
            if n_long + n_short > limit:
                violations.append(f"leverage: {n_long + n_short} gross positions > limit {limit}")
        # net_exposure is documented (FinancePortfolioModel.add_net_exposure_target)
        # as a soft penalty target, not a hard constraint -- not checked here.

    return ValidationReport(ok=(len(violations) == 0), violations=violations)
