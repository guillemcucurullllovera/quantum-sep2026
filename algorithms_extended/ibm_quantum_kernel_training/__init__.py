"""Adapted from IBM's `prototype-quantum-kernel-training` toolkit (Apache-2.0,
Copyright 2021 IBM and its contributors). See THIRD_PARTY_NOTICES.md for exactly what
was adapted vs. built fresh, and LICENSES/ibm_quantum_kernel_training_APACHE-2.0.txt
for the full upstream license text this module's derivative-work notice refers to.

This module is deliberately scoped narrowly against existing
`algorithms.pqk` kernel infrastructure (4 feature maps, KTA gradient-ascent
training already covering the Hubregtsen et al. 2022 alignment technique) --
it adds the two things that infrastructure did not have: a *covariant*
(group-structure-aware) feature map family, and an SPSA-based training loop
as a genuinely different optimization strategy from the existing central-
difference/ADAM implementation.
"""
