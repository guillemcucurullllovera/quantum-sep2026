"""Barrier *payoff* estimation (extends the barrier *probability* comparator
already in algorithms.qae_iqae.barrier_circuits.build_A_barrier).

Architecture matches that module's existing pattern (per-index
multi-controlled gate on a StatePreparation'd index register) but rotates a
payoff ancilla by an amplitude-encoded, barrier-masked payoff angle instead
of flipping an indicator ancilla — i.e. this answers "what is the discounted
expected payoff of a call option that knocks out if S_T crosses the
barrier", not just "what is P(S_T crosses the barrier)".

LIMITATION, stated plainly per this platform's no-overclaiming convention:
the barrier here is checked only at maturity (S_T vs. barrier), matching
what a discretized-grid single-register QAE construction can represent at
small qubit counts. A true continuously- or discretely-monitored barrier
(checked at every path step) is NOT what this circuit prices — see
`monte_carlo_terminal_only` (apples-to-apples parity target) vs.
`monte_carlo_path_monitored` (the realistic price, shown for contrast) in
the validation tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from algorithms.qae_iqae.finance_integrands import bs_call_price
from algorithms_extended.qiskit_finance_extensions.uncertainty_models import DiscretizedLogNormal

BarrierKind = Literal["up_and_out", "down_and_out"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


@dataclass(frozen=True)
class BarrierPayoffIntegrand:
    kind: BarrierKind
    strike: float
    barrier: float
    dist: DiscretizedLogNormal
    grid: tuple
    probs: tuple
    mask_alive: tuple  # True where the option is still alive (not knocked out) at S_T
    payoff_angle: tuple  # per-index RY angle (0 where knocked out or payoff is 0)
    norm_h: float
    exact_grid_price: float
    monte_carlo_terminal_only: float
    monte_carlo_path_monitored: float
    monte_carlo_path_stderr: float


def _flip_or_rotate_on_index(qc, j: int, n: int, target: int, theta: float | None):
    """Apply an X-masked multi-controlled gate on `target`, conditioned on the
    index register == j. theta=None -> CX/MCX (flip); theta=angle -> controlled RY."""
    from qiskit.circuit.library import RYGate

    bits = [(j >> b) & 1 for b in range(n)]
    flipped = [q for q in range(n) if bits[q] == 0]
    for q in flipped:
        qc.x(q)
    if theta is None:
        if n == 1:
            qc.cx(0, target)
        else:
            qc.mcx(list(range(n)), target)
    elif abs(theta) > 1e-12:
        # annotated=False forces eager synthesis into a concrete multi-
        # controlled-gate decomposition. The new Qiskit default
        # (annotated=True) instead returns an AnnotatedOperation, which
        # transpiles fine for a plain simulator but made Aer's noisy
        # AerSimulator.from_backend(fake_eagle/fake_heron) path blow past
        # max_memory_mb even for a 2-3-qubit circuit (found via the
        # hardware-execution smoke test) — almost certainly because the
        # annotated wrapper forced a whole-backend-sized dense fallback
        # rather than a local gate-by-gate simulation.
        gate = RYGate(theta).control(n, annotated=False)
        qc.append(gate, list(range(n)) + [target])
    for q in flipped:
        qc.x(q)


def build_barrier_payoff_integrand(
    kind: BarrierKind,
    *,
    s0: float = 100.0,
    strike: float = 100.0,
    barrier: float,
    r: float = 0.04,
    sigma: float = 0.20,
    t: float = 0.25,
    n: int = 4,
    n_mc_paths: int = 200_000,
    n_mc_steps: int = 50,
    seed: int = 42,
) -> BarrierPayoffIntegrand:
    """kind='up_and_out': knocked out if S_T >= barrier (barrier above spot).
    kind='down_and_out': knocked out if S_T <= barrier (barrier below spot)."""
    dist = DiscretizedLogNormal.from_sigma_window(s0=s0, r=r, sigma=sigma, t=t, num_qubits=n)
    grid = dist.grid()
    probs = dist.probabilities()
    disc = math.exp(-r * t)

    if kind == "up_and_out":
        alive = tuple(s < barrier for s in grid)
    elif kind == "down_and_out":
        alive = tuple(s > barrier for s in grid)
    else:
        raise ValueError(f"unknown barrier kind {kind!r}")

    raw_payoffs = [max(s - strike, 0.0) if a else 0.0 for s, a in zip(grid, alive)]
    h = max(raw_payoffs) or 1.0
    g = [_clip01(p / h) for p in raw_payoffs]
    angles = tuple(2.0 * math.asin(math.sqrt(x)) for x in g)

    exact_grid_price = disc * h * sum(p * x for p, x in zip(probs, g))

    mc_terminal, mc_path, mc_stderr = _monte_carlo_barrier_prices(
        kind=kind, s0=s0, strike=strike, barrier=barrier, r=r, sigma=sigma, t=t,
        n_paths=n_mc_paths, n_steps=n_mc_steps, seed=seed,
    )

    return BarrierPayoffIntegrand(
        kind=kind, strike=strike, barrier=barrier, dist=dist, grid=tuple(grid), probs=tuple(probs),
        mask_alive=alive, payoff_angle=angles, norm_h=h, exact_grid_price=exact_grid_price,
        monte_carlo_terminal_only=mc_terminal, monte_carlo_path_monitored=mc_path,
        monte_carlo_path_stderr=mc_stderr,
    )


def build_A_barrier_payoff(integrand: BarrierPayoffIntegrand):
    """Build the A-circuit: index-register StatePreparation + per-index
    controlled RY rotating the payoff ancilla. Amplitude estimation on the
    payoff ancilla recovers a = sum_j p_j * g_j; price = disc * H * a."""
    from qiskit import QuantumCircuit

    n = integrand.dist.num_qubits
    qc = QuantumCircuit(n + 1, name=f"A_barrier_payoff_{integrand.kind}")
    qc.append(integrand.dist.state_preparation(), list(range(n)))
    anc = n
    for j, theta in enumerate(integrand.payoff_angle):
        if theta is not None and abs(theta) > 1e-12:
            _flip_or_rotate_on_index(qc, j, n, anc, theta)
    return qc


def price_from_amplitude(integrand: BarrierPayoffIntegrand, a: float) -> float:
    disc = math.exp(-integrand.dist.r * integrand.dist.t)
    return disc * integrand.norm_h * a


def _monte_carlo_barrier_prices(
    *, kind: BarrierKind, s0: float, strike: float, barrier: float, r: float, sigma: float, t: float,
    n_paths: int, n_steps: int, seed: int,
) -> tuple[float, float, float]:
    """Returns (terminal-only price, path-monitored price, path-monitored stderr)."""
    from classical.monte_carlo.gbm import gbm_paths

    paths = gbm_paths(s0=s0, mu=r, sigma=sigma, t=t, n_steps=n_steps, n_paths=n_paths, seed=seed)
    terminal = paths[:, -1]
    disc = math.exp(-r * t)

    if kind == "up_and_out":
        alive_terminal = terminal < barrier
        alive_path = (paths < barrier).all(axis=1)
    else:
        alive_terminal = terminal > barrier
        alive_path = (paths > barrier).all(axis=1)

    payoff_t = disc * alive_terminal * (terminal - strike).clip(min=0)
    payoff_p = disc * alive_path * (terminal - strike).clip(min=0)
    return float(payoff_t.mean()), float(payoff_p.mean()), float(payoff_p.std(ddof=1) / (n_paths**0.5))


def vanilla_reference_price(s0: float, strike: float, r: float, sigma: float, t: float) -> float:
    """Unbarriered vanilla call price — an upper bound the barrier price must never exceed."""
    return bs_call_price(s0, strike, r, sigma, t)
