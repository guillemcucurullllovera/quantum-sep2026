# Release validation status

Validation is divided between local deterministic checks, isolated-wheel
evidence and the complete CI run. Reported evidence does not establish a
historical market result or practical quantum advantage.

- Source union: three authorized refs; `SOURCE_COVERAGE.json` lists every
  inspected source path, selected destination/hash, superseded revision and
  exact exclusion reason. `SOURCE_COVERAGE.md` is the source-to-publication
  mapping.
- Local release verifier: **PASS**, 1,969 retained source artifacts hash
  verified and Python source parsed; third-party notices preserved.
- Seeded synthetic release smoke: **PASS** for QAE barrier-grid oracle, QAOA
  norm/cardinality, fidelity-kernel PSD, QRC finite readout, VQC v1/v4/v5
  normalization and three MILP/enumeration cross-checks.
- Current isolated installed-wheel suite: **103 passed, 0 failed/skipped**;
  package imports were confirmed from the isolated environment's `site-packages`,
  dependency check passed and the seeded synthetic smoke passed against the wheel.
  The suite exercises source families rather than claiming full historical
  replication.
- Additional scalar-likelihood oracle tests: **12 passed** in the wheel suite;
  they compare every vectorized MLAE grid likelihood/minimum against the original
  scalar implementation.
- Current staged-source scalar-likelihood oracle tests: **12 passed**.
- First GitHub broad-suite attempt: **1,928 passed, 86 failed, 37 skipped, 8
  deselected, 1 xfailed**. The failures were classified: 80 assertions required
  source-era result/report inputs deliberately excluded from this release, five
  tested for retaining prohibited retired investor artifacts, and one noisy
  Ramsey fit was nondeterministic under an implicit simulator seed. The test
  scope and deterministic fix are now in the release; affected focused tests
  pass (**112 passed, 80 explicit skips**). Full CI rerun remains pending.
- Checkpointed broad source suite: **in progress**. Completed individual cases
  are recorded durably in the planning workspace at
  `data/derived/release_checks/quantum-parent-test-checkpoints.jsonl`; only
  complete setup/call/teardown passes are counted. No final suite total is
  claimed here until its worker exits successfully.
- CI is configured to run the offline suite, wheel build/install/smokes and a
  checksum-verified complete Git-history secret scan. CI runs without provider
  credentials; the hardware integration code is included in the source release.
  Local/CI evidence uses simulators, synthetic inputs, fake backends or mocks; no
  real QPU execution is reported or validated. Test evidence is scoped to the
  specific procedures recorded above.

Excluded tests require private datasets, credentials, provider accounts,
network access or historically frozen empirical result archives. Exclusions
and their source paths are itemized in the coverage matrix; they are not
reported as passes.
