# VQC/QML Version History

This directory preserves the actual source for every materially distinct VQC
generation built over the project's history, alongside unified
`ansatz.py`/`training.py`/`encoding.py`/`calibration.py` abstraction at the
top level (which generalizes v1/v4/v5 as selectable variants for production
use). The `v1/`–`v5/` subdirectories below are the real historical
implementations, ported with import-path fixes only — no rewriting.

| Version | Source | Qubits | Params | Origin |
|---|---|---|---|---|
| v1 | `v1/circuit.py` (`VQCSmokeTest`) | 8 | 8 | Gen1, Phase 1A |
| v2 | `v2/circuit_v2.py` (`TrainableVQC`) | 8 | 25 | Gen1, Phase 1B |
| v3 | `v3/circuit_v3.py` (`TrainableVQC_v3`) | 8 | 33 | Gen1, Phase 2 |
| v4 | `v4/circuit_v4.py` + `v4/circuit_v4_qiskit.py` | 14 (configurable) | 57 | Gen1, Phase 3B |
| v5 | `v5/ansatz.py` + `layerwise_trainer.py` + `vqc_v5.py` | 10 (configurable) | 40 (n=10,d=2) | Gen2, Phase 3B redesign |

## v1 — Phase 1A smoke test
Pure-NumPy, 8 qubits, angle encoding (`RY(feature × π)`) + linear CNOT chain +
one variational RY layer, single-qubit `⟨Z₀⟩` measurement. Proves the
encoding pipeline works end-to-end; not a trained classifier. Parameter-shift
gradients supported but not exercised in a training loop.

## v2 — Phase 1B trainable
Upgrades to circular (ring) CNOT entanglement so every parameter can affect
the measurement, two variational layers, and a weighted-sum-of-`⟨Z⟩` readout
through a sigmoid — the first version that's actually a trainable binary
classifier. Adds a hand-rolled Adam optimizer. 25 parameters (16 variational
+ 8 output weights + 1 bias).

## v3 — Phase 2, data re-uploading
Three changes over v2, each independently motivated: (1) **data
re-uploading** — features are re-encoded between variational layers, not
just at the start, which the source docstring cites as "the single biggest
improvement for quantum classifiers on small datasets"; (2) **learned
encoding scales** — each feature gets a trainable scale instead of a fixed
`× π`, so the circuit learns how much weight to give each input; (3) **cross
entanglement** between qubit pairs chosen to reflect known financial
correlations (momentum↔vol, carry↔risk-sentiment, etc.), not just adjacent
qubits. 33 parameters.

## v4 — Phase 3B, 14-qubit 1:1 feature mapping
Scales from 8 to 14 qubits — one qubit per feature in the full QIV feature
vector, eliminating the lossy 14→8 compression the earlier versions required.
Domain-aware cross-entanglement pairs are re-derived for the full 14-feature
set. Architecture becomes fully configurable (`n_qubits`, `n_layers` as
constructor params, not module globals) to support future scaling. Ships
with a parity-tested Qiskit twin (`circuit_v4_qiskit.py`) — verified in this
port to agree with the NumPy reference to `<1e-6`. 57 parameters.
Benchmark target documented in the source: >72% directional accuracy
(v3 baseline: 69.7% live / 52.7% validation average).

## v5 — Phase 3B redesign, five barren-plateau fixes (Gen2)
v4 failed in production with two independently diagnosed problems: a barren
plateau (`Var[∂f/∂θ] ≈ 2⁻¹⁴`, confirmed by a gradient-variance probe) and
overparameterization (`γ = 28 params / 16,384 training-set dim ≈ 8.5e-4`).
v5 is a from-scratch redesign applying five fixes, each grounded in a cited
theorem (documented in the source): **(1)** 10 qubits instead of 14
(McClean 2018 threshold scaling); **(2)** σ=0.01 near-identity parameter
initialization (Cerezo 2021); **(3)** layerwise pretraining — train layer 0
alone first, then unfreeze (Cerezo 2021, avoids the global plateau early in
training); **(4)** `d=2` data re-uploading via a richer Fourier-spectrum
argument (Jerbi 2023 Theorem 2, Schuld Theorem 5.1); **(5)** closed-form
ridge-regression readout instead of a trained linear layer, justified by
Schuld's Representer Theorem — the output layer becomes convex, removing one
more source of plateau.

**Important caveat, preserved from the source and already reflected in this
platform's docs**: the Q10 classical-simulability audit (2026-07-11) found
v5, like PQK-v2 and QRC-10, is classically simulable (CSIM). Its own module
docstring states this explicitly: beating a classical baseline with v5 is
*not* evidence of quantum advantage, and the "Level 2" go/no-go bar built
around that comparison was formally withdrawn. Level 1 (beat a logistic-
regression baseline, 63.8% OOS accuracy, as an exploratory-QPU gate) remains
the only claim v5 supports. This platform carries that caveat forward rather
than the superseded framing — see `docs/` for the full audit.

## What's *not* a separate preserved version
No `experimental/` directory was created: nothing found in `phase0/scripts/`
represented a materially distinct VQC *architecture* beyond v1–v5 — the
VQC-specific scripts found there (`vqc_smoke_test.py`, `test_v4_correctness.py`)
are experiment/validation harnesses for the versions above, not new
architectures, and are preserved under `prototypes/vqc/` instead, per the
"don't create empty/artificial versions" rule.
