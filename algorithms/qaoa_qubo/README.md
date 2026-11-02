# QAOA / QUBO

Two generations of portfolio-QAOA implementation are kept side by side because
they represent real, documented development history, not redundant copies:

- **`gen1/`** — the original NumPy statevector simulator implementation.
 Ring-XY mixer, random-restart optimization, amplitude amplification
 post-processing. Achieved ~39% of the brute-force-optimal Sharpe ratio on
 cardinality-constrained portfolio instances.
- **`gen2/`** — a later implementation, which diagnosed *why* Gen1
 underperformed (ring topology has no relationship to the asset correlation
 structure it's optimizing over) and fixed it with two changes: SK
 mean-field warm-start angles instead of random initialization, and a
 correlation-based complete-XY mixer instead of a ring. Documented target:
 70-80% of brute-force-optimal Sharpe, vs Gen1's 39%.

`gen2` is the recommended default for new work — `solvers.py` uses it unless
told otherwise — but `gen1` is kept runnable both as a historical baseline
and because its amplitude-amplification post-processing step has no Gen2
equivalent yet.

## Net-new platform layer

- `qubo_builder.py` — `PortfolioQUBOBuilder` / `LongShortQUBOBuilder`: a
 constraint suite (cardinality, turnover, transaction costs, sector limits,
 liquidity penalties, long/short with leverage limits, a simplified risk-
 budget penalty) that neither generation had — both only supported a bare
 cardinality constraint.
- `solvers.py` — a unified `QAOASolver` interface with two backends: the
 fast NumPy Gen2 engine, and a real Qiskit `QAOAAnsatz` + primitives path
 (simulator or noisy fake-backend) for hardware-aware compilation.
- `classical_baseline.py` — exact brute force (n<=20) and simulated
 annealing, so every QAOA result has an honest comparator.

## What's still missing (see the build report for the full list)

MILP reference solver (e.g. via a real mixed-integer solver rather than SA),
QPU execution against real IBM hardware (interface exists in
`hardware/ibm_runtime`, not yet wired end-to-end here), a rigorous
tracking-error/concentration-limit implementation (currently only
approximated by the equal-weight cardinality structure), noisy-simulation
sweep experiments (single noisy runs are supported, systematic sweeps are
not yet built).
