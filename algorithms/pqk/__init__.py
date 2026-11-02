"""Projected Quantum Kernel (PQK) suite.

Two generations, both kept:

- Gen1 (`feature_maps.py`, `kernels_gen1.py`): the original fidelity/projected
quantum kernel pipeline built in the Gen1 research repository, with three
  feature maps (AngleCNOT, ZZFeatureMap, CPMap) and a from-scratch SMO SVM.
  Its ZZFeatureMap is now known (see kta_alignment module docstring) to
  suffer kernel concentration — kept as a documented historical baseline,
  not the recommended path for new work.

- Gen2 (`pqk_v2.py`, `kta_alignment.py`, `nystrom.py`): the fix — angle
  encoding + CNOT ring (avoids the concentration failure mode), trained via
  kernel-target-alignment gradient ascent, with a Nyström approximation for
  O(m) inference instead of O(N^2) full Gram matrix computation. This is the
  recommended entry point: `from algorithms.pqk.pqk_v2 import PQKv2`.

New in this consolidation: `kernels.py` (unified fidelity/projected Gram
computation with PSD correction), `caching.py` (content-addressed kernel
matrix cache), `training.py` (registry-logged training/sweep pipeline),
`noise_study.py` (noiseless-vs-fake-backend sensitivity study).
"""
