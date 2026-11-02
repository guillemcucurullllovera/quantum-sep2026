# Release validation status

Validation is deliberately divided between local deterministic checks, the
isolated-wheel evidence, and the complete CI run. None of these establishes a
historical market result or real-hardware advantage.

- Source union: three authorized refs; `SOURCE_COVERAGE.json` lists every
  inspected source path, selected destination/hash, superseded revision and
  exact exclusion reason. `SOURCE_COVERAGE.md` is the source-to-publication
  mapping.
- Local release verifier: **PASS**, 1,969 retained source artifacts hash
  verified and Python source parsed; third-party notices preserved.
- Seeded synthetic release smoke: **PASS** for QAE barrier-grid oracle, QAOA
  norm/cardinality, fidelity-kernel PSD, QRC finite readout, VQC v1/v4/v5
  normalization and three MILP/enumeration cross-checks.
- Isolated installed-wheel evidence: **103 passed, 0 failed/skipped** in the
  previously recorded wheel suite; the tested runtime Python sources were
  byte-identical to the staged source suite. The suite exercises source
  families rather than claiming full historical replication.
- Additional scalar-likelihood oracle tests: **12 passed** in the recorded
  wheel evidence; they compare every vectorized MLAE grid likelihood/minimum
  against the original scalar implementation.
- Current staged-source scalar-likelihood oracle tests: **12 passed**.
- Checkpointed broad source suite: **in progress**. Completed individual cases
  are recorded durably in the planning workspace at
  `data/derived/release_checks/quantum-parent-test-checkpoints.jsonl`; only
  complete setup/call/teardown passes are counted. No final suite total is
  claimed here until its worker exits successfully.
- CI is configured to run the offline suite, wheel build/install/smokes and a
  checksum-verified complete Git-history secret scan. No provider credentials
  are injected. IBM/Braket hardware readiness code is retained, but actual QPU
  execution is not claimed or validated.

Excluded tests require private datasets, credentials, provider accounts,
network access or historically frozen empirical result archives. Exclusions
and their source paths are itemized in the coverage matrix; they are not
reported as passes.
