"""qrbf-OCSVM — replication of Pranjić et al. (arXiv:2411.16970), *Unsupervised
Quantum Anomaly Detection on Noisy Quantum Processors*.

ADAPTED COMPONENT — see THIRD_PARTY_NOTICES.md in this directory before modifying.

Paper's Table 1 (r=0.2, test F1, 10-fold CV, simulator column):
    rbf-OCSVM      0.757 ± 0.062
    qrbf-CX        0.829 ± 0.043   (quantum beats classical by +7.2pp)
    qrbf-ECR       0.847 ± 0.059   (quantum beats classical by +9.0pp)
    qrbf-RXX       0.827 ± 0.054   (quantum beats classical by +7.0pp)

HONEST STATUS (do not silently "improve" this without updating
REPLICATION_REPORT.md and this docstring together): a from-scratch
reimplementation against a freshly generated Sparkov dataset, following the
paper's Appendix A/B as closely as its prose description allows (no reference
code was ever published by the authors), reproduces neither the magnitude nor
the *direction* of the paper's central claim:

    rbf      : 0.507 ± 0.108   (paper 0.757 -- -25.0pp)
    qrbf-CX  : 0.197 ± 0.131   (paper 0.829 -- -63.2pp)
    qrbf-ECR : 0.152 ± 0.107   (paper 0.847 -- -69.5pp)
    qrbf-RXX : 0.066 ± 0.123   (paper 0.827 -- -76.1pp)

Root cause (diagnosed, not a bug in this port -- see REPLICATION_REPORT.md
section 3): the qrbf kernel saturates near 1.0 (mean off-diagonal similarity
0.909 vs 0.107 for classical RBF at the same gamma) -- the technique as
specified is structurally starved of discriminative signal on this dataset.
Ruled out: 1Q-RDM extraction bugs, feature-map construction bugs,
hyperparameter mis-setting (grid-searched), dataset scale, wrong OCSVM
library, wrong subsampling protocol. Not ruled out (paper publishes no code):
the authors' exact unpublished feature-engineering recipe and the specific
stochastic Sparkov realization they used.

This module is included as real, working, well-tested research
infrastructure documenting a genuine negative result -- not as a validated
anomaly-detection capability. `GATE_PASSED` below is asserted False; any
future work that claims this technique "now works" must update both this
docstring and REPLICATION_REPORT.md with the evidence, not just flip the
constant.
"""

N_QUBITS = 10
TOLERANCE_PP = 0.05  # +/- 5 percentage points, the paper's own reproducibility bar

PAPER_TABLE_1 = {
    "rbf":      (0.757, 0.062),
    "qrbf-CX":  (0.829, 0.043),
    "qrbf-ECR": (0.847, 0.059),
    "qrbf-RXX": (0.827, 0.054),
}

MEASURED_TEST_F1 = {
    "rbf":      (0.507, 0.108),
    "qrbf-CX":  (0.197, 0.131),
    "qrbf-ECR": (0.152, 0.107),
    "qrbf-RXX": (0.066, 0.123),
}

GATE_PASSED = False
assert GATE_PASSED is False, (
    "This constant documents a real, diagnosed replication failure (see "
    "REPLICATION_REPORT.md). If you have new evidence the technique reproduces, "
    "update MEASURED_TEST_F1 with real measured numbers, update "
    "REPLICATION_REPORT.md with your methodology, and only then flip this flag."
)
