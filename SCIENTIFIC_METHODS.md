# Scientific methods and experiment map

## Amplitude estimation and finance

`algorithms/qae_iqae` builds discrete expected-payoff and crash-barrier oracles,
iterative amplitude estimation and noise-aware likelihood experiments. Its adapted
SpinQIT estimator support remains separately MIT-attributed, unlike the dependency-
free narrow QAE export. `experiments/quantum1b/qae_vs_mc.py` and `run_track_e_*`
compare query/shot/resource accounting with classical sampling. A quadratic oracle
query theorem does not imply practical speedup after state preparation, simulation,
noise, loading and classical QMC/importance-sampling competitors.

## Optimization

`algorithms/qaoa_qubo` retains the original QUBO construction and both QAOA
generations, cardinality-preserving mixers and warm-start variants. The research
branch adds explicit candidate-instance generators, objective/constraint encoding,
brute-force/MILP exact competitors, heuristic gates and distribution-aware CVaR
QAOA. Relevant modules are `experiments/quantum1b/{classical_exact,qubo_encoding,
cvar_qaoa,classical_competitor_gate,mixer_warmstart_bench}.py`; seeded runners
are retained for every identified track. Optimality checks must compare feasible
solutions against the same decoded instance, not merely a low QUBO penalty value.

## Learning, kernels and reservoirs

`algorithms/pqk` supplies feature maps, projected/fidelity kernels, alignment,
Nyström approximations and cache/noise tools. `algorithms/vqc_qml` preserves v1–v5
circuits, encodings, training and layerwise strategies. `algorithms/qrc` preserves
quantum state/observable readout and reservoir math. Applications add fold-local
preprocessing, calibration, rolling evaluation, regime/anomaly and volatility tasks.
`experiments/quantum1b/qrc_esn_harness.py` compares quantum-reservoir models to ESN
competitors using common synthetic tasks. A favorable seed is not an advantage
claim, and restricted historical finance data is not replaced by invented data.

## Hardware and evidence

IBM/Braket adapters, fake-backend parity, transpilation, readout mitigation, ZNE,
PEC, dynamical decoupling and resource estimates are retained. The research branch
adds explicit budgets, retries, persistent spending state, frozen synthetic
controls, evidence lineage, authorization gates and command-center exports.
Mock and fake-backend tests exercise software behavior under the stated test
conditions. Provider-specific execution, private attestation and operational
safety require their own documented validation and authorization. Apply explicit
resource budgets and access controls for hardware workflows.

## References and attribution

See original scientific module docstrings and upstream notices. Method references
include Brassard et al. (2002), amplitude amplification/estimation; Grinko et al.
(2021), iterative QAE; Farhi et al. (2014), QAOA; Barkoutsos et al. (2020), CVaR
optimization; Havlíček et al. (2019), quantum feature spaces; Huang et al. (2021),
power of data; Fujii & Nakajima (2017), quantum reservoirs; Temme, Bravyi & Gambetta
(2017), ZNE; and Pranjić et al., arXiv:2411.16970, quantum-RBF anomaly detection.
References identify methods, not authorship of the upstream library implementations.
