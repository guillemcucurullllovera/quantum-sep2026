"""Gradient-free / natural-gradient optimizers adapted from PennyLane's
``pennylane.optimize`` (Apache-2.0, Xanadu Quantum Technologies Inc.).

See ``THIRD_PARTY_NOTICES.md`` (section) for the honest verdict per
optimizer: ``RotosolveOptimizer`` combines one small verbatim-portable
scipy-only helper (the iterative brute-force univariate substep optimizer)
with a from-scratch reimplementation of the published closed-form update
rule for two-eigenvalue generators (Ostaszewski et al. 2021). The rest of
``pennylane/optimize/`` (``QNGOptimizer``, ``QNSPSAOptimizer``,
``AdaptiveOptimizer``, plain ``GradientDescentOptimizer`` and its
momentum/Adam/RMSProp variants) is tightly coupled to
``pennylane.workflow.QNode`` / ``pennylane.gradients.metric_tensor`` /
``pennylane._grad`` (Autograd tracing) and was judged not cleanly portable
-- see the manifest for the specific import chains checked.
"""
from algorithms_extended.pennylane_asymmetrica.optimize.rotosolve import RotosolveOptimizer

__all__ = ["RotosolveOptimizer"]
