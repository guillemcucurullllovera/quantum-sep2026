# Modified from upstream: only the CPU (numpy_vectorized) precomputation path
# is wired here; gpu_numba's CUDA path was not ported (no GPU in this
# environment; see THIRD_PARTY_NOTICES.md).
from .numpy_vectorized import precompute_vectorized_cpu_parallel
