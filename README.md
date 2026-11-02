# Quantum SEP2026

Comprehensive quantum-finance research platform, including the original algorithm
implementations, finance applications, classical competitors, experiment runners,
hardware-readiness gates and tests. This parent is separate from the published
QAE, QAOA-cardinality and signal-replication standalone packages.

## Reproduce offline

```sh
uv sync --locked --extra dev --python 3.12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync pytest -q -m 'not slow'
uv run --no-sync python tools/verify_release.py
uv build
```

Use a dedicated environment: generic original namespaces such as `core`,
`algorithms` and `applications` are preserved for compatibility.

## Scientific architecture

| Layer | Contents |
|---|---|
| `algorithms` | QAE/IQAE, QUBO/QAOA with cardinality/XY mixers, projected kernels, quantum reservoirs, VQC/QNN v1–v5 |
| `applications` | Portfolio constraints, options/barriers/Greeks, tail risk, regime/anomaly/classification and time-series research |
| `classical` | Exact MILP/brute-force, annealing, Monte Carlo/QMC, ESN, HMM, kernel and financial baselines |
| `experiments/quantum1b` | CVaR-QAOA, competitor gates, QRC–ESN, QAE–MC, mixer/warm-start, noise validation and sequential runners |
| `hardware` | IBM/Braket adapters, fake backends, mitigation, transpilation, resource estimation and explicit credential gates |
| `core` | Typed schemas, reproducible seeds, registries, provenance, evidence and verification |
| `benchmarks`, `examples`, `prototypes` | Original research drivers and compatibility generations, not historical advantage endorsements |
| `algorithms_extended` | Independently implemented literature methods **and separately attributed permissively licensed upstream/adapted code** |

See SOURCE_COVERAGE.json for every retained/excluded source path and branch/commit.
THIRD_PARTY_NOTICES.md and LICENSES/ preserve upstream Apache-2.0/MIT attribution:
third-party implementations are **not first-party portfolio authorship**. Retaining
these permitted upstream dependencies avoids stripping first-party finance bridges
and research integrations that use them. No restricted third-party material ships.

## Boundaries

No credentials, private market observations, live QPU job records, production
configuration or historical result archives are included. Real hardware execution
requires your own credentials, budget and authorization. Offline tests use synthetic,
ideal/noisy-simulator, fake-backend or mocked-service inputs; they do not establish
historical financial performance or practical quantum advantage.

Original first-party source attribution: Asymmetrica Investments and Francisco.
Authorized parent export: Guillem Cucurull Llovera. Root Apache-2.0 applies only to
authorized first-party material; upstream licenses remain unchanged. See
PROVENANCE.md, LIMITATIONS.md, DEPENDENCIES.md and TEST_RESULTS.md.
