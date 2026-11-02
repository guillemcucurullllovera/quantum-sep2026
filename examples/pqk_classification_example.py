"""End-to-end example: PQKv2 (kernel-based) vs. classical RBF-SVM regime
classification on a synthetic dataset. Run:
    python -m examples.pqk_classification_example
"""
from __future__ import annotations

from applications.classification.kernel_classifier import run_kernel_classification_comparison


def main() -> None:
    result = run_kernel_classification_comparison(n_samples=80, n_splits=2, seed=0)
    print("PQKv2 vs. classical RBF-SVM regime classification:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
