"""Classical optimization baselines: every quantum optimization method in
this platform (QAOA/QUBO first and foremost) must be benchmarkable against
at least the brute-force exact solver (small instances) and simulated
annealing (larger instances); MILP is offered where the problem is linear.
"""
from classical.optimization.brute_force import brute_force_qubo
from classical.optimization.simulated_annealing import simulated_annealing_qubo
from classical.optimization.milp_solver import milp_binary_qp

__all__ = ["brute_force_qubo", "simulated_annealing_qubo", "milp_binary_qp"]
