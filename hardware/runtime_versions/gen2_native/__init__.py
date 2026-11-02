"""
gen2_native/ — Qiskit Runtime V2 wrappers with injectable backend (Gen2, historical).

This is the
original Gen2 IBM Runtime layer, preserved alongside — not replacing — the
newer unified `hardware.ibm_runtime`/`hardware.primitives` abstraction.
See README.md in this directory for which one to use for new work.

Backend injection contract:
 mode="statevector" -> StatevectorSimulator (no noise; fast dev)
 mode="fake_heron" -> FakeHeronV2 (noise model; pre-QPU validation)
 mode="ibm_boston" -> live Heron r3 (requires IBM credentials via
 core.config — NEVER read raw os.environ here,
 see backend_factory.py)

Key design: QUANTUM_BACKEND env var controls mode. Same code, no changes.
When IBM access is granted: set QUANTUM_BACKEND=ibm_boston and pull the live
backend.coupling_map to regenerate qubit subgraph indices (see
hardware.transpilation / hardware.qubit_selection for that logic — the
`subgraph_selector` concept is represented there, not duplicated here).

Components: backend_factory, estimator_wrapper, sampler_wrapper, noise_learner.

Reference: quantum_layer/study_notes/S1_ibm_platform_qiskit.md
"""
