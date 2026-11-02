"""Probabilistic Error Cancellation (PEC).

Internal Asymmetrica code. Provides overhead/feasibility analysis for
quasi-probability-based error cancellation; unbiased but more expensive
than ZNE, used when an algorithm needs higher precision than ZNE achieves.

References: van den Berg et al. (2023); Temme et al. (2017).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pec_one_norm(two_qubit_gate_error_rate: float, n_two_qubit_gates: int) -> float:
    """One-norm overhead gamma for `n_two_qubit_gates` depolarizing-noise
    two-qubit gates at the given error rate: gamma = (1 + 2*lambda/(1-lambda))^n
    """
    if two_qubit_gate_error_rate <= 0:
        return 1.0
    gamma_gate = 1 + 2 * two_qubit_gate_error_rate / (1 - two_qubit_gate_error_rate)
    return gamma_gate ** n_two_qubit_gates


def pec_required_shots(target_std_error: float, gate_error_rate: float, n_gates: int) -> int:
    gamma = pec_one_norm(gate_error_rate, n_gates)
    return int(np.ceil((gamma / target_std_error) ** 2))


def quasi_probability_decompose(gate_error_rate: float) -> tuple[np.ndarray, float]:
    """Simplified quasi-probability decomposition of an ideal 2-qubit gate
    under a depolarizing noise model: leading identity + depolarizing terms.
    """
    lam = gate_error_rate
    q_ideal = 1.0 / (1 - lam)
    q_flip = -lam / (1 - lam)
    quasi_probs = np.array([q_ideal, q_flip])
    return quasi_probs, float(abs(q_ideal) + abs(q_flip))


@dataclass
class PECMitigator:
    gate_error_rate: float = 0.024
    n_two_qubit_gates: int = 45

    def overhead(self) -> float:
        return pec_one_norm(self.gate_error_rate, self.n_two_qubit_gates)

    def required_shots(self, target_std: float = 0.01) -> int:
        return pec_required_shots(target_std, self.gate_error_rate, self.n_two_qubit_gates)

    def check_feasibility(self, target_std: float = 0.01, max_shots: int = 10_000) -> dict:
        gamma = self.overhead()
        n_req = self.required_shots(target_std)
        return {
            "feasible": n_req <= max_shots,
            "overhead": round(gamma, 4),
            "required_shots": n_req,
            "budget_shots": max_shots,
            "gate_error_rate": self.gate_error_rate,
            "n_two_qubit_gates": self.n_two_qubit_gates,
        }

    def summary(self) -> dict:
        return {
            "method": "Probabilistic Error Cancellation (van den Berg 2023)",
            "gate_error_rate": self.gate_error_rate,
            "n_two_qubit_gates": self.n_two_qubit_gates,
            "one_norm_overhead": round(self.overhead(), 4),
            "required_shots_at_0.01_std": self.required_shots(0.01),
        }


__all__ = ["PECMitigator", "pec_one_norm", "pec_required_shots", "quasi_probability_decompose"]
