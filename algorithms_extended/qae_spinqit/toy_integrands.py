"""Toy integrands (upstream MIT benchmark set) — the reproducible g0/g1/g2/x/x^2
functions used to validate the QAE pipeline before any finance integrand is
plugged in. Adapted (MIT) from spinqit-qae-triangulum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from algorithms_extended.qae_spinqit.quadrature import Rule, grid_points


@dataclass(frozen=True)
class Integrand:
    name: str
    kind: str  # "toy" | "finance" | "finance-barrier"
    label: str
    g: Callable[[float], float]
    exact_integral: Callable[[float], Optional[float]]
    meta: Dict[str, object] = field(default_factory=dict)


def _clip01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def theta_from_value(v: float) -> float:
    """Angle map theta = 2*arcsin(sqrt(v)), v in [0,1] -> theta in [0,pi]."""
    return 2.0 * math.asin(math.sqrt(_clip01(v)))


TOY: Dict[str, Integrand] = {
    "g0": Integrand(
        name="g0", kind="toy", label="1/4",
        g=lambda x: 0.25,
        exact_integral=lambda y: 0.25 * y,
        meta={"degree": 0, "a_exact_uniform": 0.25},
    ),
    "g1": Integrand(
        name="g1", kind="toy", label="sin^2(pi*x/2)",
        g=lambda x: math.sin(math.pi * x / 2.0) ** 2,
        exact_integral=lambda y: 0.5 * y - math.sin(math.pi * y) / (2.0 * math.pi),
        meta={"degree": 1, "a_exact_uniform": 0.5},
    ),
    "g2": Integrand(
        name="g2", kind="toy", label="sin^2(pi*x)",
        g=lambda x: math.sin(math.pi * x) ** 2,
        exact_integral=lambda y: 0.5 * y - math.sin(2.0 * math.pi * y) / (4.0 * math.pi),
        meta={"degree": 2, "a_exact_uniform": 0.5},
    ),
    "x": Integrand(
        name="x", kind="toy", label="x",
        g=lambda x: _clip01(x),
        exact_integral=lambda y: 0.5 * y ** 2,
        meta={"degree": 1},
    ),
    "x2": Integrand(
        name="x2", kind="toy", label="x^2",
        g=lambda x: _clip01(x) ** 2,
        exact_integral=lambda y: y ** 3 / 3.0,
        meta={"degree": 2},
    ),
}


def get_toy_integrand(name: str) -> Integrand:
    if name not in TOY:
        raise KeyError(f"unknown toy integrand {name!r}; choices={list(TOY)}")
    return TOY[name]
