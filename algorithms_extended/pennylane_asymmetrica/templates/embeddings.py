"""``AngleEmbedding`` and ``QAOAEmbedding``, ported from PennyLane's
``pennylane/templates/embeddings/{angle,qaoaembedding}.py``
(Apache-2.0, Xanadu Quantum Technologies Inc.).

Same porting rationale as ``templates/layers.py``: only the
``compute_decomposition`` pure gate-sequence logic was carried across,
re-targeted at Qiskit's ``QuantumCircuit``.

``QAOAEmbedding``'s "local field" Ising-ansatz layer uses PennyLane's
``MultiRZ`` two-qubit case (the decomposition only ever calls it pairwise,
on wires ``[i, i+1]``): ``MultiRZ(theta, wires=[a,b]) = exp(-i*theta/2 * Z_a
Z_b)``, which is *exactly* Qiskit's ``rzz(theta, a, b)`` gate -- verified
against both projects' docstrings, no adjustment needed.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from core.errors import ValidationError

_ROTATION_GATES = {"X": "rx", "Y": "ry", "Z": "rz"}


def angle_embedding(
    qc: QuantumCircuit,
    features: np.ndarray,
    wires: list[int] | None = None,
    rotation: str = "X",
) -> QuantumCircuit:
    """Port of ``AngleEmbedding.compute_decomposition``.

    Encodes N <= len(wires) features as single-qubit rotations, one feature
    per wire; wires beyond ``len(features)`` are left untouched (matching
    upstream: PennyLane's constructor truncates ``wires`` to ``n_features``
    rather than padding, so extra wires simply see no gate here either).

    Args:
        qc: circuit to append gates onto (mutated in place, also returned).
        features: shape ``(N,)``, ``N <= len(wires)``.
        wires: qubit indices; defaults to ``range(len(features))``.
        rotation: one of ``"X"``, ``"Y"``, ``"Z"``.
    """
    features = np.asarray(features, dtype=float).reshape(-1)
    n_features = features.shape[0]
    wires = list(wires) if wires is not None else list(range(n_features))
    if n_features > len(wires):
        raise ValidationError(
            f"angle_embedding: features must be of length {len(wires)} or less; got {n_features}"
        )
    key = rotation.upper()
    if key not in _ROTATION_GATES:
        raise ValidationError(f"angle_embedding: unknown rotation '{rotation}'")
    rot_fn = getattr(qc, _ROTATION_GATES[key])
    for i in range(n_features):
        rot_fn(float(features[i]), wires[i])
    return qc


def qaoa_embedding_shape(n_layers: int, n_wires: int) -> tuple[int, int]:
    """Port of ``QAOAEmbedding.shape`` (broadcasting case omitted -- not used here)."""
    if n_wires == 1:
        wire_dim = 1
    elif n_wires == 2:
        wire_dim = 3
    else:
        wire_dim = 2 * n_wires
    return (n_layers, wire_dim)


def qaoa_embedding(
    qc: QuantumCircuit,
    features: np.ndarray,
    weights: np.ndarray,
    wires: list[int] | None = None,
    local_field: str = "Y",
) -> QuantumCircuit:
    """Port of ``QAOAEmbedding.compute_decomposition`` (Killoran et al. 2020,
    arXiv:2001.03622): alternating feature-encoding (RX per feature, H for
    any remaining wires) and a trainable 1D-Ising-model "weight Hamiltonian"
    (MultiRZ ring / local-field rotations), repeated ``n_layers`` times, with
    one final feature-encoding pass appended after the last layer.

    Args:
        qc: circuit to append gates onto (mutated in place, also returned).
        features: shape ``(N,)``, ``N <= len(wires)``.
        weights: shape ``(n_layers, wire_dim)`` per ``qaoa_embedding_shape``.
        wires: qubit indices; defaults to ``range(len(wires_inferred))`` --
            must be given explicitly if ``len(features) < intended n_wires``.
        local_field: one of ``"X"``, ``"Y"``, ``"Z"`` (upstream default "Y").
    """
    features = np.asarray(features, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValidationError(f"qaoa_embedding: weights must be 2D; got shape {weights.shape}")
    n_layers, wire_dim = weights.shape

    if wires is None:
        # Infer n_wires from the weight-tensor width per upstream's shape formula.
        if wire_dim in (1, 3):
            n_wires = wire_dim if wire_dim == 1 else 2
        else:
            n_wires = wire_dim // 2
        wires = list(range(n_wires))
    else:
        wires = list(wires)
    n_wires = len(wires)
    n_features = features.shape[0]
    if n_features > n_wires:
        raise ValidationError(
            f"qaoa_embedding: features must be of length {n_wires} or less; got {n_features}"
        )

    key = local_field.upper()
    if key not in _ROTATION_GATES:
        raise ValidationError(f"qaoa_embedding: unknown local_field '{local_field}'")
    field_fn_name = _ROTATION_GATES[key]

    exp_wire_dim = 1 if n_wires == 1 else (3 if n_wires == 2 else 2 * n_wires)
    if wire_dim != exp_wire_dim:
        raise ValidationError(
            f"qaoa_embedding: weights width must be {exp_wire_dim} for n_wires={n_wires}; got {wire_dim}"
        )

    def _encode() -> None:
        for i in range(n_features):
            qc.rx(float(features[i]), wires[i])
        for i in range(n_features, n_wires):
            qc.h(wires[i])

    for layer in range(n_layers):
        _encode()
        w = weights[layer]
        field_fn = getattr(qc, field_fn_name)
        if n_wires == 1:
            field_fn(float(w[0]), wires[0])
        elif n_wires == 2:
            qc.rzz(float(w[0]), wires[0], wires[1])
            field_fn(float(w[1]), wires[0])
            field_fn(float(w[2]), wires[1])
        else:
            for i in range(n_wires):
                qc.rzz(float(w[i]), wires[i], wires[(i + 1) % n_wires])
            for i in range(n_wires):
                field_fn(float(w[n_wires + i]), wires[i])

    _encode()
    return qc
