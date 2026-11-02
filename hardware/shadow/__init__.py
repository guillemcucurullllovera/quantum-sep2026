"""Shadow deployment infrastructure.

This is the bridge between QPU/simulator training and
low-latency classical production inference: extract quantum features once,
approximate them classically (Nyström), and export frozen weights for
sub-10ms inference without a QPU in the loop.

See hardware/shadow/shadow_kernel.py for how ShadowKernel differs
from algorithms/pqk/caching.py's KernelCache — they solve different
problems and are both kept.
"""
