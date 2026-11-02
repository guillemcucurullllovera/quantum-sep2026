"""The Quantum Lab experiment registry.

A read-only, hand-curated index of every QUANTUM-1B experiment that has
actually run, built entirely from existing docs/quantum/*.md and
experiments/quantum1b/results/*.json -- no new science, no new numbers.
Every field below is sourced from a specific document/result file (see
`source` on each record); this module does not recompute or re-derive any
finding.

**Why hand-curated, not auto-generated from the docs:** the per-track
markdown documents are prose, written for a human reader, with corrections
and retractions interleaved inline (see the P1-2/P1-6/P1-7/P1-8
remediation history in `docs/quantum/QUANTUM1B_CVAR_QAOA.md` and
`QUANTUM1B_HARDWARE_READINESS.md`) -- parsing that reliably and
mechanically would be its own significant, fragile project. Instead, each
`QuantumLabExperiment` below was transcribed directly from the CURRENT
(post-P1-remediation) text of its source document by a dedicated
extraction pass, cross-checked against the underlying result JSON file
paths. If a track's document changes, this module can silently drift --
exactly the kind of gap `frozen_controls.py` had before P1-5 fixed it. A
lightweight freshness check is NOT implemented here (unlike
`frozen_controls.py`'s snapshot-vs-recomputation check, there is no cheap
way to mechanically re-derive prose); treat this module's content as a
snapshot as of 2026-10-01 and re-verify by reading the source docs before
relying on it for anything safety- or disclosure-relevant.
"""
from __future__ import annotations

from dataclasses import dataclass

from experiments.quantum1b.quantum_lab.readiness_state import (
    ReadinessGates,
    ReadinessState,
    compute_readiness_state,
    current_lab_wide_gates,
)


@dataclass(frozen=True)
class QuantumLabExperiment:
    experiment_id: str
    track: str
    problem: str
    hypothesis: str
    classical_baseline: str
    quantum_formulation: str
    simulator_noise_config: str
    result: str
    evidence_level: str
    conclusion: str
    limitations: str
    next_experiment: str
    result_artifact_paths: tuple[str, ...]
    source_doc: str
    has_ideal_simulator_result: bool
    has_noise_validated_result: bool
    hardware_readiness_pathway_exists: bool
    """See `readiness_state.ReadinessGates.hardware_readiness_pathway_exists`
    -- True only for Tracks A/B/D, which share the `OptionSelectionInstance`/
    QUBO lineage `frozen_controls.py` and the Hardware Day runbook are built
    for. Track C (QRC/ESN) and Track E (QAE/IQAE) use different circuit
    representations with no comparable frozen-control/runbook pathway built
    yet."""

    def readiness_state(self) -> ReadinessState:
        lab_wide = current_lab_wide_gates()
        gates = ReadinessGates(
            has_classical_baseline=True,  # every record below cites one explicitly, enforced by test_registry.py
            has_ideal_simulator_result=self.has_ideal_simulator_result,
            has_noise_validated_result=self.has_noise_validated_result,
            hardware_readiness_pathway_exists=self.hardware_readiness_pathway_exists,
            **lab_wide,
        )
        return compute_readiness_state(gates)


QUANTUM_LAB_REGISTRY: tuple[QuantumLabExperiment, ...] = (
    QuantumLabExperiment(
        experiment_id="EXP-A1",
        track="A",
        problem="Certify the classical MILP comparator (Fortet-linearized MIQP) is exact before trusting it as Track B's ground truth.",
        hypothesis="Fortet-linearized MILP agrees exactly with brute-force enumeration across all tested instances.",
        classical_baseline="solve_exact_enumeration (vectorized 2^n brute force) vs. solve_exact_milp (scipy.optimize.milp/HiGHS)",
        quantum_formulation="none -- pure classical correctness gate",
        simulator_noise_config="n/a -- no quantum execution",
        result="360/360 cells (9 families x n in {4..18} x 5 seeds) bit-exact agreement, max abs diff 0.0. Runtime 28.7s (budget <5min).",
        evidence_level="A (preregistered static label); computed via core.evidence.compute_evidence_level = B -- documented discrepancy, both values recorded in the result JSON (compute_evidence_level has no reachable level-A branch for a pure-analytical classical result)",
        conclusion="PASS -- classical comparator certified exact. A correctness gate, not a scientific claim.",
        limitations="compute_evidence_level's missing level-A branch for analytical results is a flagged gap in core/evidence.py, out of this track's scope to fix.",
        next_experiment="none stated -- feeds directly into EXP-A2/Track B as the certified ground truth",
        result_artifact_paths=("experiments/quantum1b/results/track_a_results.json",),
        source_doc="docs/quantum/QUANTUM1B_CLASSICAL_BENCHMARK.md",
        has_ideal_simulator_result=False,  # pure classical, no quantum simulator run
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=True,
    ),
    QuantumLabExperiment(
        experiment_id="EXP-A2",
        track="A",
        problem="Measure the classical MILP comparator's scaling/tractability ceiling vs. Track B's reach.",
        hypothesis="READY iff every n<=16 solves exactly within a 120s/cell budget.",
        classical_baseline="solve_exact_milp, family=dense (worst-case O(n^2) Fortet growth), n in {8,12,16,20,24}, seeds 0-2",
        quantum_formulation="none",
        simulator_noise_config="n/a",
        result="15/15 cells solved exactly (HiGHS status 0) up to n=24 -- the ladder ceiling, exceeding the required n<=16. Runtimes 13ms(n=8)-31ms(n=24); total 0.3s (budget <30min). Simulated-annealing fallback never triggered.",
        evidence_level="A (preregistered); same compute_evidence_level discrepancy as EXP-A1 applies",
        conclusion="READY -- classical comparator classified READY, exceeds the requirement with substantial margin.",
        limitations="none disclosed beyond the evidence-level computation gap shared with EXP-A1",
        next_experiment="none stated",
        result_artifact_paths=("experiments/quantum1b/results/track_a_results.json",),
        source_doc="docs/quantum/QUANTUM1B_CLASSICAL_BENCHMARK.md",
        has_ideal_simulator_result=False,
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=True,
    ),
    QuantumLabExperiment(
        experiment_id="EXP-B1",
        track="B",
        problem="Does CVaR-QAOA (risk-averse objective aggregation) beat standard expectation-QAOA on objective gap vs. the exact classical optimum, on the candidate-trade / option-selection QUBO?",
        hypothesis="CVaR-QAOA shows a significant objective-gap or feasible-probability improvement over expectation-QAOA (preregistered Wilcoxon signed-rank test).",
        classical_baseline="classical_exact.solve_exact_milp (Track A, evidence level A) on the identical OptionSelectionInstance",
        quantum_formulation="Custom NumPy statevector-exact standard RX-mixer QAOA (cvar_qaoa.py), COBYLA-driven, p in {1,2}, alpha in {1.0,0.5,0.2}",
        simulator_noise_config="Exact NumPy statevector simulation, no noise, no shots (ideal). 6 families x 4 sizes (n in [4,6,8,10]) x p x alpha x 5 seeds = 720 runs.",
        result="CVaR-best-alpha and expectation-QAOA median-of-medians objective gap both = 0.0. CVaR improves-or-ties in 24/24 cells. Wilcoxon signed-rank: uncomputable (all-zero diff vector). Feasible-prob>=0.5 in 5/24 cells. Runtime 255.4s (budget <4h).",
        evidence_level="B",
        conclusion="RESEARCH_ONLY (pre-committed classification) -- both variants already hit the classical optimum at every tested size, so no gap exists for the significance test to detect. CVaR shows a modest, non-significant feasible-probability improvement (+0.011 median). No quantum-advantage claim.",
        limitations="Ladder reduced from the preregistered n in [4..14] to n in [4,10] (logged preregistration deviation -- the exact slack-bit QUBO needs more qubits/candidate than assumed, pushing n>=12 past the statevector engine's practical ceiling).",
        next_experiment="EXP-B2 (shot sensitivity on the winning cell), EXP-B3 (CSIM reproduction)",
        result_artifact_paths=("experiments/quantum1b/results/track_b_exp_b1.json",),
        source_doc="docs/quantum/QUANTUM1B_CVAR_QAOA.md",
        has_ideal_simulator_result=True,
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=True,
    ),
    QuantumLabExperiment(
        experiment_id="EXP-B2",
        track="B",
        problem="How does CVaR estimation quality degrade as finite shot count decreases, on EXP-B1's winning cell?",
        hypothesis="CVaR estimation error decreases monotonically as shots increase (descriptive, no pass/fail gate).",
        classical_baseline="exact CVaR computed from the full discrete distribution",
        quantum_formulation="same cvar_qaoa.py engine, EXP-B1's winning cell (family=independent, n=4, p=1, alpha=0.5, seed 0)",
        simulator_noise_config="finite-shot sampling at shots in {256,1024,4096,16384}, 10 seeds each, no device noise",
        result="mean|CVaR_shots - CVaR_exact|: 256->0.05132, 1024->0.03696, 4096->0.01814, 16384->0.00877 (monotonically decreasing). Runtime sub-minute (budget <15min).",
        evidence_level="C",
        conclusion="Confirms the hypothesis -- monotonic shot-error decay. Descriptive only, no pass/fail gate.",
        limitations="none disclosed",
        next_experiment="none stated",
        result_artifact_paths=("experiments/quantum1b/results/track_b_exp_b2.json",),
        source_doc="docs/quantum/QUANTUM1B_CVAR_QAOA.md",
        has_ideal_simulator_result=True,
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=True,
    ),
    QuantumLabExperiment(
        experiment_id="EXP-B3",
        track="B",
        problem="Is CVaR-QAOA classically simulable (CSIM) at the sizes reachable here, by the same Q10-audit methodology (provenance + structural/entanglement) already applied to PQK-v2/QRC-10/VQC-v5?",
        hypothesis="Reproduce the Q10 audit's two-route CSIM logic (provenance + bounded-entanglement structural check) for CVaR-QAOA.",
        classical_baseline="n/a -- classification exercise, not a pass/fail comparator study (the economic comparison is EXP-B1's, both QAOA variants tie the exact optimum, never beat it)",
        quantum_formulation="cvar_qaoa.py statevector simulation, family=dense (EXP-A2's worst-case pairwise-risk family), candidate ladder (4,6,8,10,12,14), p in {1,2}",
        simulator_noise_config="Exact NumPy statevector (no noise); n_qubits up to 23 (~134MB statevector); theoretical memory-crossover n_qubits=29 (analytical, not reached)",
        result="CSIM_BY_PROVENANCE_AT_EVERY_N_TESTED. Entanglement growth across 12 cells: schmidt_rank from 12/16 (n=9) up to 256/256 saturated (n=16,p=2) down to 32/2048 (n=23,p=1, fixed-angle regime) -- no clean bounded-chi saturation signature. Runtime 359.8s (budget 1200s).",
        evidence_level="A (structural/provenance classification) + B (the diagnostic runs themselves)",
        conclusion="CSIM_BY_PROVENANCE_AT_EVERY_N_TESTED -- consistent with the Q10 audit's now-4-for-4 CSIM finding across every variational/kernel method tested in this estate. No bounded-chi classical surrogate justified. No quantum-advantage claim at any n tested or at the originally-targeted untested n_qubits=25.",
        limitations=(
            "The original 16-candidate/25-qubit cell was dropped after TWO separate host-OOM incidents. "
            "The FIRST was root-caused with high confidence (an O(dim*n) memory bug in cost_diagonal_from_ising, "
            "fixed, independently verified bit-identical). A SECOND, separate OOM was hit even after that fix, "
            "when re-attempting the 25-qubit ceiling, and was NOT conclusively diagnosed (most likely candidate: "
            "BLAS thread-local workspace / malloc-arena fragmentation under concurrent host load, not a repeat of "
            "the same bug) -- the ladder was trimmed to 14 candidates/23 qubits as a precaution. This is the "
            "corrected account after P1-6 remediation (2026-10-01): the original documentation overstated "
            "certainty here ('found and fixed, not merely worked around', a traceback 'confirmed directly' -- "
            "both retracted)."
        ),
        next_experiment="none stated beyond 'a real Hardware Day prep pass would benefit from more points'",
        result_artifact_paths=("experiments/quantum1b/results/track_b_exp_b3.json",),
        source_doc="docs/quantum/QUANTUM1B_CVAR_QAOA.md",
        has_ideal_simulator_result=True,
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=True,
    ),
    QuantumLabExperiment(
        experiment_id="EXP-C1",
        track="C",
        problem="Does QuantumReservoir (QRC) beat a dimension-matched classical Echo State Network (ESN) on a synthetic GARCH(1,1)-style volatility-proxy forecasting task?",
        hypothesis="A properly matched classical ESN ties or beats QuantumReservoir (preregistered).",
        classical_baseline="EchoStateNetwork, readout_dim matched to QRC's n1+n2 qubit count; persistence and ridge-on-raw-lags baselines also reported",
        quantum_formulation="QuantumReservoir (algorithms/qrc/reservoir.py), readout dimension D=n1+n2 (only diagonal Z-expectation populations reach the readout, not the full 4^n2-dim density matrix)",
        simulator_noise_config="exact density-matrix simulation, no shot/device noise; D in {4,6,8} (reduced from preregistered {4,8,16} -- D=16 would need a ~68.7GB matrix, hard infeasibility), 5 seeds, n=2000-point series",
        result="ESN beats QRC on 15/15 (D,seed) cells on out-of-sample MSE. Paired t-test (ESN-QRC) at every D: p<0.001. Dambre-2012 memory-capacity diagnostic: ESN 2.1-2.9 total MC vs. QRC 0.038-0.062 (ESN ~50-75x higher linearly-decodable memory capacity). QLIKE confirms the MSE finding in all 15 cells. Runtime 171.4s (budget <10min).",
        evidence_level="B",
        conclusion="NEGATIVE for QRC -- confirms the preregistered hypothesis. Matched-dimension classical ESN beats QuantumReservoir on both MSE and linear memory capacity, every D tested, statistically significant. Consistent with the existing Q10 audit's QRC-10 finding.",
        limitations=(
            "RESEARCH_ONLY is the maximum reachable classification from this result regardless of outcome -- real-"
            "financial-task execution is explicitly BLOCKED in code (RealFinancialExecutionBlocked exception) "
            "pending Model Lab's data-licensing hold (no confirmed-rights OHLCV for the AMC underlyings). The "
            "QRC near-zero memory capacity is not attributed to an implementation bug (checked directly) but to "
            "the reservoir's nonlinear readout being a poor match for a strictly linear probe."
        ),
        next_experiment="Unblock condition: Model Lab issuing a RESEARCH_ONLY authority contract covering licensed daily OHLCV for the AMC underlyings (would still only reach evidence B/C even once unblocked)",
        result_artifact_paths=("experiments/quantum1b/results/track_c_results.json",),
        source_doc="docs/quantum/QUANTUM1B_QRC_ESN.md",
        has_ideal_simulator_result=True,
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=False,  # no frozen-control/runbook pathway exists for QRC's circuit type
    ),
    QuantumLabExperiment(
        experiment_id="EXP-D1",
        track="D",
        problem="How does real-device-calibrated noise (FakeKingston) affect QAOA circuit depth/gate count and feasibility vs. an ideal simulator, and how does this scale with problem size, on the SAME candidate-trade QUBO Track B uses?",
        hypothesis="Not a formal preregistered hypothesis -- Track D is infrastructure/engineering-readiness work, not a hypothesis-driven science track.",
        classical_baseline="instance.evaluate_selection (the true constrained objective, never the QUBO's own claimed cost) decodes both ideal and noisy cells identically",
        quantum_formulation="algorithms.qaoa_qubo.solvers._build_qiskit_qaoa_circuit (standard-mixer QAOA, matching cvar_qaoa.py's convention), p=1, COBYLA",
        simulator_noise_config=(
            "Ideal: bare AerSimulator(). Noisy: AerSimulator.from_backend(FakeKingston()) (real 156-qubit device "
            "calibration snapshot, full coupling map + gate/readout error). optimization_level=1 (qiskit default) "
            "-- CORRECTED 2026-10-01 (P1-2): the original claim of a sharp transpile-cost threshold at n_qubits=14 "
            "requiring optimization_level=0 did NOT reproduce across 5 independent reproductions; "
            "optimization_level=1 completes in ~3s and is SHALLOWER (depth~375) than optimization_level=0 "
            "(depth~486) at that cell."
        ),
        result=(
            "n=3 (9 qubits): depth 14->176 (12.6x), 2-qubit gates 28->116 (4.1x), feasible-fraction ~flat "
            "(0.485->0.464), objective gap 0.0 both, wall-clock 5.9s->69.4s (11.8x). COMPLETED. "
            "n=6 (12 qubits): depth 20->408 (20.4x), fake_kingston runtime 594.7s. COMPLETED. "
            "n=8 (14 qubits): NOT_COMPLETED -- two bounded noisy-COBYLA attempts (30min, 45min) did not finish; "
            "the real bottleneck is noisy-execution wall-clock time, NOT transpilation (that earlier claim is retracted)."
        ),
        evidence_level="C (ideal) / E (FakeKingston-noisy)",
        conclusion=(
            "2 completed points originally showed the depth-inflation ratio growing with n (12.6x -> 20.4x) -- "
            "CORRECTED framing (P2-13): initially downgraded to 'directionally suggestive, NOT a statistically "
            "established trend' (only 2 single-seed, unreplicated points). **UPDATED 2026-10-01: a dedicated "
            "5-seeds-per-rung replication (run_track_d_exp_d1_seed_replication.py) now CONFIRMS the growth is "
            "real** -- mean ratio 12.77 (std=0.91, n=5 seeds) at n_candidates=3/9 qubits vs. 18.73 (std=1.18, "
            "n=5 seeds) at n_candidates=6/12 qubits, Mann-Whitney U one-sided p=0.004 (significant at p<0.05). "
            "This replication transpiles only (no COBYLA optimization loop -- depth is a function of circuit "
            "structure, not optimized angles, so this measures the identical quantity far more cheaply). The "
            "originally-claimed 'sharp threshold at n_qubits=14' and 'optimization_level=0 is the fix' remain "
            "RETRACTED (P1-2) -- this replication is about the SEPARATE depth-inflation-vs-n finding, not that "
            "retracted claim. What actually caused the original >8.5-minute transpile observation remains "
            "genuinely unknown."
        ),
        limitations=(
            "The n=8 cell's attempt-timestamp record is internally inconsistent (P1-7); the '>45 minutes, direct "
            "evidence' framing does not survive this and should be read as 'not completed within a bounded, "
            "imprecisely-timed attempt.' The depth-inflation-vs-n finding now has real seed replication (5 seeds "
            "x 2 rungs) and statistical significance, but remains only 2 rungs (9 and 12 qubits) -- not a fitted "
            "scaling law across many n, and says nothing about whether the trend continues, accelerates, or "
            "plateaus beyond n_qubits=12. The n=3 vs. n=6 WALL-CLOCK runtime comparison (separate from the depth "
            "finding) still uses different maxiter (15 vs. 10) in the original single-seed measurement -- not a "
            "strictly controlled comparison."
        ),
        next_experiment=(
            "PARTIALLY DONE 2026-10-01: run_track_d_exp_d1_n8_single_shot_probe.py attempted to isolate what "
            "bounds the n_qubits=14 noisy-execution cell, without re-attempting the full open-ended COBYLA "
            "loop that failed twice before. Confirmed backend build (~11s) and transpilation (~2.6s, depth=376, "
            "matching the P1-2 calibration almost exactly) are fast and NOT the bottleneck. The single noisy "
            "execution itself did NOT complete within two independently bounded attempts (280s, then 90s) -- "
            "giving a real lower bound (>280s per evaluation) rather than resolving the exact cost, which "
            "remains genuinely unmeasured. A real Hardware Day prep pass would still benefit from more EXP-D1 "
            "ladder rungs (beyond n_qubits=12) to fit an actual scaling law."
        ),
        result_artifact_paths=(
            "experiments/quantum1b/results/track_d_exp_d1.json",
            "experiments/quantum1b/results/track_d_exp_d1_ladder.json",
            "experiments/quantum1b/results/track_d_exp_d1_n8.json",
            "experiments/quantum1b/results/track_d_exp_d1_seed_replication.json",
            "experiments/quantum1b/results/track_d_exp_d1_n8_single_shot_probe.json",
        ),
        source_doc="docs/quantum/QUANTUM1B_HARDWARE_READINESS.md",
        has_ideal_simulator_result=True,
        has_noise_validated_result=True,
        hardware_readiness_pathway_exists=True,
    ),
    QuantumLabExperiment(
        experiment_id="EXP-E1",
        track="E",
        problem="Standing classical-competitor gate -- does amplitude estimation's theoretical edge survive known classical competitors (QMC, importance sampling, Grover-Rudolph loading cost) on the default Black-Scholes bull-call-spread/crash-barrier book?",
        hypothesis="Three pre-committed, fixed-threshold checks (not tuned post-hoc).",
        classical_baseline="QMC (quasi-Monte Carlo), importance-sampling variance reduction, Grover-Rudolph state-loading cost scaling (erosion_audit.py, reused not reimplemented)",
        quantum_formulation="n/a -- the gate evaluates classical-competitor erosion against AE's theoretical advantage; no quantum circuit run here",
        simulator_noise_config="n/a -- deterministic classical computation",
        result="QMC convergence edge: FAIL (zeta_qmc=-1.614, threshold <=-0.9). Importance-sampling erosion: FAIL (173.3x variance-reduction, threshold >=10x). Grover-Rudolph loading cost: FAIL (ae_slope=1.027, gr_slope=1.000, both in [0.8,1.2]). Overall verdict: FAIL.",
        evidence_level="A (deterministic classical computation, no execution uncertainty)",
        conclusion="FAIL -- all three classical competitors erode AE's edge on this specific book; the gate correctly blocks any new AE proposal against this book without a different integrand or novel classical-competitor argument.",
        limitations="Specific to the default book (Black-Scholes bull-call-spread/crash-barrier) -- a different integrand could pass.",
        next_experiment="none stated -- a standing, reusable precondition gate for future AE proposals",
        result_artifact_paths=("experiments/quantum1b/results/track_e_exp_e1.json",),
        source_doc="docs/quantum/QUANTUM1B_TRACK_E_AE_VS_MC.md",
        has_ideal_simulator_result=False,
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=False,  # no frozen-control/runbook pathway exists for AE's circuit type
    ),
    QuantumLabExperiment(
        experiment_id="EXP-E2",
        track="E",
        problem="Does IQAE beat classical Monte Carlo at a MATCHED query budget on the crash-barrier integrand, under ideal (noiseless) simulation?",
        hypothesis="IQAE's RMSE is lower than budget-matched classical MC's RMSE at tight epsilon, statistically significant.",
        classical_baseline="classical_mc_budget_matched (same total query count as IQAE's measured usage, n=16,247) -- added as a methodological fix after the original comparison used an unmatched 2048-sample classical MC (~8x fewer queries than IQAE)",
        quantum_formulation="IQAE (algorithms/qae_iqae/iqae.py::run_iqae), crash-barrier integrand P(S_T<=95%S_0), T=0.25, epsilon=0.01 (tight, real Grover amplification), alpha=0.05, 30 trials",
        simulator_noise_config="ideal statevector/shot simulation, no device noise",
        result="IQAE: 16,247 queries/trial, bias=-0.00002, RMSE=0.0006, CI coverage=1.00, runtime=0.49s. Classical MC matched (n=16,247): bias=+0.0013, RMSE=0.0033, coverage=1.00. Mann-Whitney U (matched): p=1.9e-5.",
        evidence_level="B (ideal statevector/shot simulation, no device noise)",
        conclusion="At matched budget, IQAE's RMSE is ~5.5x lower than classical MC's, statistically significant (p=1.9e-5). Directionally/roughly quantitatively consistent with the theoretical quadratic AE speedup -- but this is a SINGLE measurement at one (epsilon,N) point, not a scaling-law fit, so it SUPPORTS rather than conclusively proves the asymptotic claim.",
        limitations="Single (epsilon,N) point, not a scaling sweep; ideal simulation only (noise sensitivity addressed separately in EXP-E3); no quantum-advantage claim made. (An earlier, unmatched 2048-sample classical MC comparison is kept in the result record for transparency but is methodologically invalid and not used for this conclusion.)",
        next_experiment="EXP-E3 (same comparison under FakeKingston noise)",
        result_artifact_paths=("experiments/quantum1b/results/track_e_exp_e2.json",),
        source_doc="docs/quantum/QUANTUM1B_TRACK_E_AE_VS_MC.md",
        has_ideal_simulator_result=True,
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=False,
    ),
    QuantumLabExperiment(
        experiment_id="EXP-E3",
        track="E",
        problem="Does IQAE's ideal-simulator precision advantage (EXP-E2) survive realistic device noise, even in the simplest-possible circuit (zero Grover amplification)?",
        hypothesis="Not a directional preregistered hypothesis -- tests noise sensitivity; epsilon deliberately loosened to 0.05 so IQAE needs zero amplification (the best case for IQAE under noise).",
        classical_baseline="classical MC, shots=2048 (genuinely budget-matched here since oracle_queries=0 means shots IS the comparable query count)",
        quantum_formulation="IQAE, same crash-barrier integrand, epsilon=0.05, FakeKingston noise, 10 trials",
        simulator_noise_config="FakeKingston-noisy (NoiseModel.from_backend(FakeKingston())), shots=2048, 0 Grover amplification rounds",
        result="IQAE: bias=+0.0195, RMSE=0.0278, CI coverage=0.60, runtime=5.10s. Classical MC: bias=-0.0038, RMSE=0.0133, coverage=0.80. Mann-Whitney U (RMSE): p=0.104 (not significant at n=10). A disclosed, unresolved non-determinism exists in FakeKingston's noise-model construction under a nominally-fixed seed (a repeat run gave different numbers).",
        evidence_level="E (FakeBackend device-calibrated noise)",
        conclusion="IQAE's ideal-simulator precision advantage does NOT reliably survive realistic device noise, even in the simplest circuit. RMSE degradation is numerically worse but NOT statistically confirmed at n=10 (p=0.104) -- reported as 'numerically worse, not statistically confirmed', not overstated as a settled negative. CI-coverage collapse (30-60% vs. 95% nominal) IS clearly established -- the track's most concrete negative finding.",
        limitations="Only 10 trials (noisy runs are expensive, per Track B's own OOM-incident precedent); FakeKingston noise-model non-reproducibility under a nominally-fixed seed is disclosed, not resolved; the specific 30-60% CI figures are this book's own numbers, not a universal constant.",
        next_experiment="A larger noisy-trial count would be needed to formally confirm/rule out the RMSE-level claim (not done, cost-prohibitive per Track B's OOM precedent)",
        result_artifact_paths=("experiments/quantum1b/results/track_e_exp_e3.json",),
        source_doc="docs/quantum/QUANTUM1B_TRACK_E_AE_VS_MC.md",
        has_ideal_simulator_result=True,
        has_noise_validated_result=True,
        hardware_readiness_pathway_exists=False,
    ),
    QuantumLabExperiment(
        experiment_id="EXP-E4",
        track="E",
        problem="How does IQAE's Grover-amplified circuit depth scale with k (Grover iterations), and what does this imply about real-hardware feasibility at EXP-E2's actual epsilon=0.01 setting?",
        hypothesis="Not a formal preregistered hypothesis -- closes the hardware-aware-cost research-standard requirement, mirroring Track D's EXP-D1 for QAOA.",
        classical_baseline="none -- pure circuit-cost/transpilation measurement, no classical comparator",
        quantum_formulation="algorithms.qae_iqae.iqae._grover_metadata (reused, not reimplemented), crash-barrier integrand, ks=(0,1,2,4,8,16,32,64,128)",
        simulator_noise_config="FakeKingston, optimization_level=0 (the pre-existing function's own convention, reused as-is -- NOT re-justified, and explicitly NOT cross-validated against Track D's retracted finding, per the P1-2 correction)",
        result="Depth/2-qubit-gates by k: k=0->444/121 ... k=16->18000/4958 ... k=128->141216/39333. OLS fit: depth~452+1101*k (matches k=128 within ~0.2%). No transpile wall observed at any k.",
        evidence_level="E (FakeBackend device-calibrated, structural/transpilation measurement -- no shots executed)",
        conclusion="Clean linear scaling, no non-linear blow-up in this k range. EXP-E2's epsilon=0.01 run plausibly reaches k in the tens or higher in its final round -- even k=16 already implies depth=18,000, far beyond any near-term device's coherence/fidelity budget. EXP-E3's choice of epsilon=0.05/k=0 (depth=444) is likely close to the ONLY depth regime this circuit could plausibly run on a real device today.",
        limitations="Does not instrument run_iqae's internal adaptive-round schedule, so the exact deepest single circuit EXP-E2 actually built is not directly reported -- the conclusion follows from the scaling trend plus the large oracle-query total, not a pinpointed k. The original cross-reference to Track D's EXP-D1 as 'confirmation' is RETRACTED (P1-2), since that Track D finding did not reproduce -- this track's own measurement stands independently, but the cross-validation claim does not.",
        next_experiment=(
            "DONE 2026-10-01: whether optimization_level=1 does better here was tested "
            "(run_track_e_exp_e4_opt_level_comparison.py) -- optimization_level=1 is consistently SHALLOWER "
            "than optimization_level=0 across the full ks sweep (e.g. depth 107,619 vs 141,216 at k=128, "
            "~23.8% shallower), the same directional pattern found for Track B's QAOA circuit (P1-2), though "
            "measured independently here rather than re-derived from that finding. Does not change this "
            "experiment's own conclusion (both opt levels show clean linear scaling with no non-linear "
            "blow-up, and depth at any plausible k remains far beyond near-term hardware's coherence budget "
            "either way)."
        ),
        result_artifact_paths=(
            "experiments/quantum1b/results/track_e_exp_e4.json",
            "experiments/quantum1b/results/track_e_exp_e4_opt_level_comparison.json",
        ),
        source_doc="docs/quantum/QUANTUM1B_TRACK_E_AE_VS_MC.md",
        has_ideal_simulator_result=False,
        has_noise_validated_result=True,
        hardware_readiness_pathway_exists=False,
    ),
    QuantumLabExperiment(
        experiment_id="EXP-F1",
        track="F",
        problem="Does parameter-initialization strategy (cold-start-random vs. linear-ramp vs. INTERP warm-start) affect CVaR-QAOA's objective quality or optimizer efficiency on Track B's engine?",
        hypothesis="Not directional -- the calibrated expectation was a null result, given EXP-B1 already shows objective gap=0.0 at every tested size.",
        classical_baseline="n/a -- compares QAOA initialization strategies against each other; the classical optimum (Track A) is used only to compute objective gap",
        quantum_formulation="cvar_qaoa.py RX-mixer engine with an optional initial_params override; 3 strategies: cold_start_random (uniform (0,pi/2), 5 seeds), linear_ramp (deterministic adiabatic-inspired), interp (Zhou 2020, extrapolates an optimized p=1 solution to a p=2 initial guess)",
        simulator_noise_config="exact NumPy statevector, no noise; FROZEN_CONTROLS instances (independent/correlated/tight_budget, n_candidates=4, 9 qubits), maxiter=200",
        result="Every strategy at every instance reaches objective gap=0.0. COBYLA eval counts: cold_start p=1 mean 31.2(sigma 2.7); cold_start p=2 mean 39.3(sigma 3.4); interp p=2 mean 39.2(sigma 2.3) -- distributions overlap completely, no efficiency difference.",
        evidence_level="Not given its own letter in this document; implicitly follows Track B's evidence-level convention (B).",
        conclusion="Clean null finding, exactly as calibrated expectations predicted -- no warm-start strategy can improve on an already-zero objective gap, and optimizer-efficiency shows no meaningful difference either. An expected, not disappointing, outcome.",
        limitations="Tests only one regime (9-qubit instances where cold-start COBYLA already trivially finds the optimum) -- a harder regime (larger n, tighter maxiter) might show a real advantage, not tested (would carry EXP-B3's same statevector-memory risk). The XY/Dicke-mixer comparison was explicitly NOT done -- structurally incompatible with the slack-qubit QUBO encoding (a deliberate, documented non-comparison, not an oversight).",
        next_experiment="none stated; implicitly a harder/larger-n regime where cold-start might struggle",
        result_artifact_paths=("experiments/quantum1b/results/track_f_exp_f1.json",),
        source_doc="docs/quantum/QUANTUM1B_TRACK_F_MIXER_WARMSTART.md",
        has_ideal_simulator_result=True,
        has_noise_validated_result=False,
        hardware_readiness_pathway_exists=True,
    ),
)


def get_experiment(experiment_id: str) -> QuantumLabExperiment:
    for exp in QUANTUM_LAB_REGISTRY:
        if exp.experiment_id == experiment_id:
            return exp
    raise KeyError(f"no experiment {experiment_id!r} in QUANTUM_LAB_REGISTRY")


def by_track(track: str) -> tuple[QuantumLabExperiment, ...]:
    return tuple(exp for exp in QUANTUM_LAB_REGISTRY if exp.track == track)


__all__ = ["QuantumLabExperiment", "QUANTUM_LAB_REGISTRY", "get_experiment", "by_track"]
