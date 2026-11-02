# Third-Party Notices

This project incorporates source code from third-party open-source projects.
Each is used under the terms of its own license, reproduced in full under
[`LICENSES/`](LICENSES/). This document identifies which package in this
repository derives from which upstream project, its license, and the
applicable copyright notice.

Nothing in this document alters or restricts the terms of any upstream
license. Where an upstream project distributes its own `NOTICE` file
(required for Apache-2.0 attribution), that file is reproduced verbatim
under `LICENSES/`.

| Component (`algorithms_extended/…`) | Upstream project | License | License file(s) |
|---|---|---|---|
| `amazon_braket_asymmetrica/` | amazon-braket-sdk-python — Copyright 2019 Amazon.com, Inc. or its affiliates | Apache-2.0 | `amazon_braket_APACHE-2.0.txt`, `amazon_braket_NOTICE.txt` |
| `qiskit_experiments_asymmetrica/` | qiskit-experiments — Copyright 2017 IBM and its contributors | Apache-2.0 | `qiskit_experiments_APACHE-2.0.txt` |
| `qiskit_optimization_asymmetrica/` | qiskit-optimization — Copyright 2017 IBM and its contributors | Apache-2.0 | `qiskit_optimization_APACHE-2.0.txt` |
| `qiskit_ml_asymmetrica/` | qiskit-machine-learning — Copyright 2017 IBM and its contributors | Apache-2.0 | `qiskit_machine_learning_APACHE-2.0.txt` |
| `qaoa_training_pipeline_asymmetrica/` | qaoa_training_pipeline (IBM Quantum) | Apache-2.0 | `qaoa_training_pipeline_APACHE-2.0.txt` |
| `qiskit_ibm_runtime_asymmetrica/` | qiskit-ibm-runtime — Copyright 2021 IBM and its contributors | Apache-2.0 | `qiskit_ibm_runtime_APACHE-2.0.txt` |
| `pennylane_asymmetrica/` | PennyLane (Xanadu) | Apache-2.0 | `pennylane_APACHE-2.0.txt` |
| `qokit_finance/` | QOKit (JPMorgan Chase) | Apache-2.0 | `qokit_APACHE-2.0.txt` |
| `qae_spinqit/` | spinqit-qae-triangulum — Copyright (c) 2026 Antonio Falcó Montesinos | MIT | `spinqit_MIT.txt` |
| `qiskit_finance_extensions/` | qiskit-finance — Copyright 2017 IBM and its contributors | Apache-2.0 | `qiskit-finance_Apache-2.0.txt`, `qiskit-finance-data-providers_reference.txt` |
| `qiskit_ml_finance/` | qiskit-machine-learning — Copyright 2017 IBM and its contributors | Apache-2.0 | `qiskit_machine_learning_APACHE-2.0.txt` |
| `tket_asymmetrica/` | TKET / pytket-quantinuum (Quantinuum) | Apache-2.0 | `tket_APACHE-2.0.txt` |
| `qml_financial_classifiers/` | qiskit-machine-learning — Copyright 2017 IBM and its contributors | Apache-2.0 | `qiskit_machine_learning_APACHE-2.0.txt` |
| `ibm_quantum_kernel_training/` | prototype-quantum-kernel-training (qiskit-community) — Copyright 2021 IBM and its contributors | Apache-2.0 | `ibm_quantum_kernel_training_APACHE-2.0.txt` |
| `error_mitigation/` (readout mitigation) | qiskit-experiments — Copyright 2017 IBM and its contributors | Apache-2.0 | `qiskit_experiments_APACHE-2.0.txt` |
| `qae_options_qiskit_finance/` | qiskit-finance (pattern reference; no source copied) | Apache-2.0 | `qiskit-finance_Apache-2.0.txt` |
| `qubo_constraints_qiskit_optimization/` | qiskit-optimization — Copyright 2017 IBM and its contributors | Apache-2.0 | `qiskit_optimization_APACHE-2.0.txt` |

## Independent implementations of published methods

The following components implement algorithms described in published
academic literature. No upstream source code exists or was used; only the
published method is cited.

- **`error_mitigation/` (zero-noise extrapolation)** — Temme, Bravyi &
  Gambetta, "Error Mitigation for Short-Depth Quantum Circuits," *Phys.
  Rev. Lett.* 119, 180509 (2017).
- **`qaoa_initialization/`** — INTERP and linear-ramp parameter
  initialization strategies, implemented independently from the published
  literature (cross-referenced against QAOAKit, BSD-2-Clause, and the
  End-to-End QAOA Parameters protocol, Apache-2.0, for validation only;
  no source from either was incorporated).
- **`qrbf_ocsvm_anomaly/`** — Quantum-RBF-kernel one-class SVM, following
  Pranjić et al., arXiv:2411.16970. No reference implementation is
  published for this paper.

## Reviewed and not incorporated

The following projects were reviewed for potential use and are **not**
incorporated anywhere in this repository, in any form, due to
copyleft licensing incompatible with this project's terms:

- Mitiq (GPL-3.0)
- ucc (AGPL-3.0)

## Quantum reservoir computing mathematics

`algorithms/qrc/reservoir_math.py` implements standard quantum-reservoir-
computing mathematics (transverse-field Ising Hamiltonian encoding,
partial-trace readout, ridge regression) following the general architecture
described in Fujii & Nakajima (2017). It is an independent implementation
with no upstream source dependency.
