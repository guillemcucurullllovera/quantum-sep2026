"""End-to-end example: VQC directional classification vs. a classical
logistic-regression baseline. Run:
    python -m examples.vqc_qnn_classification_example
"""
from __future__ import annotations

from algorithms.vqc_qml.ansatz import VQCVersion
from applications.classification.synthetic_financial_dataset import SyntheticDirectionalDataset
from applications.classification.vqc_classifier import run_directional_classification


def main() -> None:
    dataset = SyntheticDirectionalDataset(n_samples=60, seed=2)
    results = run_directional_classification(dataset, n_qubits=4, version=VQCVersion.V5_REUPLOAD, n_splits=2)
    print("VQC (v5) directional classification, per fold:")
    for r in results:
        print(f"  fold={r.fold}  vqc_acc={r.vqc_accuracy:.3f}  classical_acc={r.baseline_accuracy:.3f}")


if __name__ == "__main__":
    main()
