"""Angle-structure (affinity) diagnostic - screen an integrand before hardware.

For the 2-index-qubit grid, the angle table theta_ij = 2*arcsin(sqrt(g(x_ij))) is
**affine** iff theta_11 = theta_10 + theta_01 - theta_00, i.e. the Moebius quadratic
coefficient c12 = theta_11-theta_10-theta_01+theta_00 is 0. The affine residual is
exactly |c12|.

  * affine (residual~=0)  -> shallow state-prep (3 gates).
  * non-affine (d=2)      -> the generic construction (adds one doubly-controlled RY).

Adapted (MIT) from spinqit-qae-triangulum `scripts/00_check_function_affinity.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from algorithms_extended.qae_spinqit.state_prep import angle_table, mobius_coeffs
from algorithms_extended.qae_spinqit.toy_integrands import Integrand


@dataclass(frozen=True)
class AffinityReport:
    integrand: str
    label: str
    rule: str
    thetas: Dict[str, float]
    c0: float
    c1: float
    c2: float
    c12: float
    affine_residual: float
    is_affine: bool
    degree: int
    classification: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "integrand": self.integrand, "label": self.label, "rule": self.rule,
            "thetas": self.thetas, "c0": self.c0, "c1": self.c1, "c2": self.c2, "c12": self.c12,
            "affine_residual": self.affine_residual, "is_affine": self.is_affine,
            "degree": self.degree, "classification": self.classification,
        }


def _classify(residual: float, tol: float) -> str:
    if residual <= tol:
        return "affine / d<=1 -- shallow state-prep (3 gates)"
    if residual <= 1e-2:
        return "near-affine -- candidate for affine compression"
    return "non-affine / d=2 -- generic construction (adds a doubly-controlled RY)"


def check_affinity(integrand: Integrand, rule: str = "midpoint", tol: float = 1e-9) -> AffinityReport:
    at = angle_table(integrand, rule=rule)
    t00, t01, t10, t11 = at[(0, 0)], at[(0, 1)], at[(1, 0)], at[(1, 1)]
    c0, c1, c2, c12 = mobius_coeffs(t00, t01, t10, t11)
    residual = abs(c12)
    is_affine = residual <= tol
    distinct = len({round(v, 12) for v in (t00, t01, t10, t11)})
    degree = 0 if distinct == 1 else (1 if is_affine else 2)
    return AffinityReport(
        integrand=integrand.name, label=integrand.label, rule=rule,
        thetas={"t00": t00, "t01": t01, "t10": t10, "t11": t11},
        c0=c0, c1=c1, c2=c2, c12=c12, affine_residual=residual,
        is_affine=is_affine, degree=degree, classification=_classify(residual, tol),
    )
