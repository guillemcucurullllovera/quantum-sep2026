# `algorithms/qae_iqae/` — module notes

This directory holds the Asymmetrica-original amplitude-estimation layer.
The upstream-derived circuit/estimator core it depends on lives in
`algorithms_extended/qae_spinqit/` (see `THIRD_PARTY_NOTICES.md`).

## Contents

- `finance_integrands.py` — Black-Scholes European bull-call-spread expected-payoff
  integrand, and the crash-barrier `P(S_T <= barrier)` construction (the Woerner-Egger
  VaR building block).
- `barrier_circuits.py` — the factored (distribution-loader + pure-comparator)
  circuit construction for the crash-barrier probability.
- `backends.py` — simulator/fake-backend wiring (ideal Aer, FakeKingston noise model).
- `mitigation.py` — M3 (mthree, Apache-2.0) readout-error mitigation on the counts path.
- `iqae.py` — orchestrates `qiskit_algorithms.IterativeAmplitudeEstimation` (Apache-2.0)
  against the crash-barrier circuit; wires results into `core.experiment_registry`.
- `runner.py` — thin CLI/API using `core` plumbing.

## Implementation notes

- **No advantage claim beyond what the literature supports.** Amplitude estimation is a
  proven *quadratic* (not exponential) speedup, fault-tolerance-gated, and eroded by
  classical QMC competitors at this platform's current low-dimensional book size. Small
  grid sizes (n=2/n=3) are pipeline/algorithm-validation demonstrations, not accurate
  prices or probabilities — every result carries this caveat in its metadata, not just
  in prose.
- **No third-party copyleft.** mthree and qiskit-algorithms are Apache-2.0. QQuantLib
  (EUPL) and Mitiq (GPL) are not used anywhere in this package.
- **`IBM_QUANTUM_TOKEN` is read exclusively via `core.config.get_settings().ibm`** —
  never hardcoded, never logged, never persisted to a result file.
