"""``BasicEntanglerLayers`` and ``StronglyEntanglingLayers``, ported from
PennyLane's ``pennylane/templates/layers/{basic_entangler,strongly_entangling}.py``
(Apache-2.0, Xanadu Quantum Technologies Inc. -- see
``LICENSES/pennylane_APACHE-2.0.txt``).

What was ported: the ``compute_decomposition`` staticmethods only -- these
are the pure gate-sequence-construction logic (weight tensor + wire list ->
flat list of gates), independent of ``pennylane.operation.Operation``,
``pennylane.wires.Wires``, capture/qjit tracing, and the ``for_loop``/``cond``
control-flow primitives the rest of each upstream file uses for
JAX-jit-compatible tracing. That control-flow machinery is not needed for
eager Python construction of a fixed-shape circuit, which is all this
platform's Qiskit-based pipeline ever does.

Gate-convention check (why no numerical adjustment was needed):
  * PennyLane RX/RY/RZ(theta)  == exp(-i*theta/2 * P)   == Qiskit rx/ry/rz(theta)
  * PennyLane Rot(phi,theta,omega) == RZ(omega) @ RY(theta) @ RZ(phi)
    (see pennylane/ops/qubit/parametric_ops_single_qubit.py, class Rot
    docstring) -- applied in circuit order RZ(phi), RY(theta), RZ(omega),
    which is exactly what ``qc.rz(phi, w); qc.ry(theta, w); qc.rz(omega, w)``
    produces (up to the same global phase convention Qiskit's own u-gate
    family uses).
  * PennyLane CNOT/CZ == Qiskit cx/cz verbatim.

Both conventions were verified with the ``StronglyEntanglingLayers.shape``
docstring numeric example carried over into
``tests/algorithms_extended/test_pennylane_templates.py``.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from core.errors import ValidationError

_ROTATION_GATES = {"rx", "ry", "rz"}


def basic_entangler_shape(n_layers: int, n_wires: int) -> tuple[int, int]:
    """Port of ``BasicEntanglerLayers.shape``: weight tensor shape (L, n_wires)."""
    return (n_layers, n_wires)


def basic_entangler_layers(
    qc: QuantumCircuit,
    weights: np.ndarray,
    wires: list[int] | None = None,
    rotation: str = "rx",
) -> QuantumCircuit:
    """Port of ``BasicEntanglerLayers.compute_decomposition``.

    One single-parameter rotation per wire per layer, followed by a ring of
    CNOTs (a single CNOT for 2 wires -- PennyLane's documented convention of
    dropping the redundant "wrap-around" CNOT on exactly 2 wires -- and no
    entangler at all for 1 wire).

    Args:
        qc: circuit to append gates onto (mutated in place, also returned).
        weights: shape ``(n_layers, len(wires))``.
        wires: qubit indices to act on; defaults to ``range(weights.shape[1])``.
        rotation: one of ``"rx"``, ``"ry"``, ``"rz"`` (PennyLane defaults to RX).
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValidationError(
            f"basic_entangler_layers: weights must be 2D (n_layers, n_wires); got shape {weights.shape}"
        )
    n_layers, n_w = weights.shape
    wires = list(wires) if wires is not None else list(range(n_w))
    if len(wires) != n_w:
        raise ValidationError(
            f"basic_entangler_layers: weights second dim {n_w} != len(wires) {len(wires)}"
        )
    rotation = rotation.lower()
    if rotation not in _ROTATION_GATES:
        raise ValidationError(f"basic_entangler_layers: unknown rotation '{rotation}'")
    rot_fn = getattr(qc, rotation)

    for layer in range(n_layers):
        for i, w in enumerate(wires):
            rot_fn(float(weights[layer, i]), w)
        if len(wires) == 2:
            qc.cx(wires[0], wires[1])
        elif len(wires) > 2:
            for i in range(len(wires)):
                qc.cx(wires[i], wires[(i + 1) % len(wires)])
    return qc


def strongly_entangling_shape(n_layers: int, n_wires: int) -> tuple[int, int, int]:
    """Port of ``StronglyEntanglingLayers.shape``: weight tensor shape (L, n_wires, 3)."""
    return (n_layers, n_wires, 3)


def strongly_entangling_layers(
    qc: QuantumCircuit,
    weights: np.ndarray,
    wires: list[int] | None = None,
    ranges: list[int] | None = None,
) -> QuantumCircuit:
    """Port of ``StronglyEntanglingLayers.compute_decomposition``.

    Per layer: a full ``Rot(phi, theta, omega)`` (= RZ.RY.RZ) on every wire,
    then a ring of CNOTs with "range" ``r`` -- CNOT(wire[i], wire[(i+r) % M])
    for every wire i. Default range schedule (when ``ranges`` is not given)
    is ``r_l = (l mod (M-1)) + 1`` for layer l, matching upstream exactly.

    Args:
        qc: circuit to append gates onto (mutated in place, also returned).
        weights: shape ``(n_layers, len(wires), 3)``.
        wires: qubit indices to act on; defaults to ``range(weights.shape[1])``.
        ranges: optional per-layer CNOT range hyperparameter, length n_layers.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 3 or weights.shape[2] != 3:
        raise ValidationError(
            f"strongly_entangling_layers: weights must have shape (n_layers, n_wires, 3); got {weights.shape}"
        )
    n_layers, n_w, _ = weights.shape
    wires = list(wires) if wires is not None else list(range(n_w))
    if len(wires) != n_w:
        raise ValidationError(
            f"strongly_entangling_layers: weights second dim {n_w} != len(wires) {len(wires)}"
        )
    m = len(wires)

    if ranges is None:
        if m > 1:
            ranges = tuple((l % (m - 1)) + 1 for l in range(n_layers))
        else:
            ranges = (0,) * n_layers
    else:
        ranges = tuple(ranges)
        if len(ranges) != n_layers:
            raise ValidationError(
                f"strongly_entangling_layers: ranges length {len(ranges)} != n_layers {n_layers}"
            )
        for r in ranges:
            if m > 0 and r % m == 0:
                raise ValidationError(
                    f"strongly_entangling_layers: range {r} must not be 0 mod n_wires={m}"
                )

    for layer in range(n_layers):
        for i, w in enumerate(wires):
            phi, theta, omega = (float(x) for x in weights[layer, i])
            qc.rz(phi, w)
            qc.ry(theta, w)
            qc.rz(omega, w)
        if m > 1:
            r = ranges[layer]
            for i in range(m):
                qc.cx(wires[i], wires[(i + r) % m])
    return qc
