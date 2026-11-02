"""P0.1 — Replication of Pranjic et al. (arXiv:2411.16970) qrbf-OCSVM on Sparkov.

Pass criterion (paper Table 1, r=0.2, 10-fold CV, test split):
    rbf-OCSVM      F1 = 0.757 ± 0.062
    qrbf-CX        F1 = 0.829 ± 0.043
    qrbf-ECR       F1 = 0.847 ± 0.059
    qrbf-RXX       F1 = 0.827 ± 0.054

Replication is considered successful if our measured means land within ±5
percentage points of the paper's reported means on each model.
"""
