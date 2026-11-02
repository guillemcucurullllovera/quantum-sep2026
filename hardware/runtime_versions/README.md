# Runtime versions

Two implementations of IBM Runtime access live here, deliberately kept side
by side per "preserve distinct generations" directive:

## `gen2_native/` — historical, component-aware

Component-aware: `EMConfig.for_component()` encodes
a specific, deliberate error-mitigation stack per algorithm (QRC gets no ZNE
because intentional decoherence is part of its signal; PQK v2 gets PEC because
kernel estimates are unusually noise-sensitive; VQC v5 and QAOA v2 get the
"standard" ZNE+DD+twirl+TREX stack). Also includes a `NoiseLearner` session-
freshness protocol (2h TTL, matching Heron r3's T1 drift window) that the
newer layer below does not have.

## `hardware/ibm_runtime/` + `hardware/primitives/` — current

A newer, general-purpose Sampler/Estimator abstraction, built from scratch,
backend-agnostic (works identically against a local Aer simulator or
a real QPU once credentials exist), and integrated with `hardware/sessions`,
`hardware/jobs`, `hardware/batching`, `hardware/retries`, and
`hardware/experiment_tracking`.

## Recommendation for new work

Use `hardware/ibm_runtime` + `hardware/primitives` for anything new — it has
the fuller job-lifecycle infrastructure (sessions, retries, batching,
provenance stamping) that `gen2_native/` doesn't. But `gen2_native/`'s
per-component EM matrix (which mitigation techniques apply to which
algorithm, and why) is a genuine piece of domain knowledge worth preserving
and consulting — if you're wiring mitigation into a specific algorithm module
in `algorithms/`, read `gen2_native/estimator_wrapper.py`'s `EMConfig.for_component`
first, then implement the equivalent decision inside the newer layer rather
than importing `gen2_native` directly into production code paths.

Both raise loudly (never silently no-op) when `qiskit_ibm_runtime` is absent
or when IBM credentials are missing — see each module's own docstring.
