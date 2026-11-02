# QAOA historical prototypes (Gen1)

Preserved research/experiment scripts from the Gen1 research track that
produced `algorithms/qaoa_qubo/gen1/`. These are development
history, not a second production implementation — algorithmic logic is
unchanged from the original; only import paths were updated to this repo's
layout.

| Script | Role |
|---|---|
| `qaoa_backtest.py` | Rolling walk-forward backtest: QAOA vs classical MV vs equal-weight |
| `phase2d_qaoa_optimize.py` | End-to-end single-run pipeline: fetch → covariance → QUBO → QAOA → compare |
| `ablation_fast.py`, `n15_overnight.py`, `n20_overnight.py`, `overnight_sweep.py` | Parameter-sweep orchestrators — each `subprocess`-invokes `qaoa_backtest.py` with different CLI configs and aggregates results |

**Not independently runnable end-to-end in `quantum_sep2026`**: the two base
scripts import `PortfolioQUBO`/`QAOACircuit`/`QAOAOptimizer` from
`algorithms.qaoa_qubo.gen1.qaoa_portfolio`, which is a real, working import —
but the original module also included `fetch_yahoo_prices` (live Yahoo
Finance network fetch), `compute_covariance_matrix` (calls that fetch), and
`adaptive_qubo_scaling` (pure math, no network dependency). None of the three
were carried into this repo's `gen1/qaoa_portfolio.py` port (deliberately dropped the network-coupled data layer). Those three names are
called only inside function bodies here — never at module import time — so
every script in this directory still imports and parses cleanly; they just
can't complete a live, data-driven run without that layer restored or
substituted. See each file's own header for the exact gap.
