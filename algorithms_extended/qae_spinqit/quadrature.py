"""Quadrature grid for the QAE numerical-integration encoding.

`n` index qubits give `2**n` grid nodes on `[0, y]`; the state-prep operator `A`
places the integrand value `g(x_i)` on the ancilla amplitude at each node, so that
`P(ancilla=1 | A|0>) = (1/2**n) * sum_i g(x_i)` (a uniform-grid quadrature of the
mean of `g`). At n=2 this is a 4-point rule — deliberately coarse; the QAE layer is a
pipeline demonstration, not an accurate quadrature (see THIRD_PARTY_NOTICES.md).

Adapted (MIT) from spinqit-qae-triangulum `src/qae/quadrature.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

Rule = Literal["left", "midpoint", "right"]


@dataclass(frozen=True)
class Grid:
    y: float
    n: int
    rule: str
    points: Tuple[float, ...]


def grid_points(y: float = 1.0, n: int = 2, rule: Rule = "midpoint") -> Grid:
    """Return the `2**n` quadrature nodes on `[0, y]` for the given rule."""
    if y <= 0:
        raise ValueError("y must be > 0")
    m = 2 ** n
    h = y / m
    if rule == "left":
        pts = tuple(i * h for i in range(m))
    elif rule == "right":
        pts = tuple((i + 1) * h for i in range(m))
    elif rule == "midpoint":
        pts = tuple((i + 0.5) * h for i in range(m))
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown rule: {rule!r}")
    return Grid(y=float(y), n=int(n), rule=str(rule), points=pts)


def index_to_bits(i: int, n: int = 2) -> Tuple[int, ...]:
    """Map grid index `i` to its `n`-bit tuple (MSB first), matching the
    affinity diagnostic convention `bits = ((i>>(n-1))&1, ..., i&1)`."""
    return tuple((i >> (n - 1 - b)) & 1 for b in range(n))
