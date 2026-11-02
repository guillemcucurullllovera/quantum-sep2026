"""Algorithm-family quantum-vs-classical benchmarks. Each submodule calls
real algorithm/classical APIs directly (see qaoa.py's module docstring for
why this bypasses the currently-non-conformant ALGORITHM_REGISTRY path) and
returns a core.schemas.BenchmarkResult via benchmarks.runner.run_benchmark.
"""
