"""Tests for algorithms_extended/pennylane_asymmetrica/templates/ -- circuit
construction logic ported from PennyLane's pennylane/templates/, re-targeted
at Qiskit's QuantumCircuit. Assertions check the ported gate sequences
against PennyLane's own documented numeric examples (docstrings quoted in
the source files) and against the claimed parameter/qubit-count contracts.
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from algorithms_extended.pennylane_asymmetrica.templates import (
    angle_embedding,
    basic_entangler_layers,
    basic_entangler_shape,
    qaoa_embedding,
    qaoa_embedding_shape,
    strongly_entangling_layers,
    strongly_entangling_shape,
)
from core.errors import ValidationError


def _z_expectations(qc: QuantumCircuit, n_qubits: int) -> np.ndarray:
    sv = Statevector(qc)
    z = np.zeros(n_qubits)
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"
        z[i] = float(sv.expectation_value(SparsePauliOp("".join(label))).real)
    return z


class TestBasicEntanglerLayers:
    def test_shape_helper(self):
        assert basic_entangler_shape(n_layers=2, n_wires=3) == (2, 3)

    def test_matches_pennylane_docstring_example(self):
        # From pennylane/templates/layers/basic_entangler.py's own docstring:
        # circuit([[pi, pi, pi]]) on 3 wires with default RX rotation and a
        # ring of CNOTs gives Z-expectations [1.0, 1.0, -1.0].
        qc = QuantumCircuit(3)
        basic_entangler_layers(qc, np.array([[np.pi, np.pi, np.pi]]), rotation="rx")
        z = _z_expectations(qc, 3)
        np.testing.assert_allclose(z, [1.0, 1.0, -1.0], atol=1e-10)

    def test_two_wires_drops_wraparound_cnot(self):
        # Upstream's documented convention: for exactly 2 wires only a single
        # CNOT is applied (no redundant wrap-around second CNOT).
        qc = QuantumCircuit(2)
        basic_entangler_layers(qc, np.array([[np.pi, np.pi]]), rotation="rx")
        cx_count = sum(1 for instr in qc.data if instr.operation.name == "cx")
        assert cx_count == 1

    def test_single_wire_no_entangler(self):
        qc = QuantumCircuit(1)
        basic_entangler_layers(qc, np.array([[np.pi]]), rotation="rx")
        assert not any(instr.operation.name == "cx" for instr in qc.data)

    def test_gate_count_scales_with_layers(self):
        n_layers, n_wires = 3, 4
        weights = np.zeros((n_layers, n_wires))
        qc = QuantumCircuit(n_wires)
        basic_entangler_layers(qc, weights, rotation="rz")
        rz_count = sum(1 for instr in qc.data if instr.operation.name == "rz")
        cx_count = sum(1 for instr in qc.data if instr.operation.name == "cx")
        assert rz_count == n_layers * n_wires
        assert cx_count == n_layers * n_wires  # full ring for n_wires > 2

    def test_rejects_shape_mismatch(self):
        qc = QuantumCircuit(3)
        with pytest.raises(ValidationError):
            basic_entangler_layers(qc, np.zeros((2, 4)), wires=[0, 1, 2])


class TestStronglyEntanglingLayers:
    def test_shape_helper(self):
        assert strongly_entangling_shape(n_layers=2, n_wires=4) == (2, 4, 3)

    def test_matches_pennylane_docstring_example(self):
        # From pennylane/templates/layers/strongly_entangling.py's own docstring:
        # weights=[[[-0.2, 0.1, -0.4], [1.2, -2., -0.4]]], ranges=(1,), wires=["a","b"]
        # decomposes to Rot(a) Rot(b) CNOT(a,b) CNOT(b,a).
        weights = np.array([[[-0.2, 0.1, -0.4], [1.2, -2.0, -0.4]]])
        qc = QuantumCircuit(2)
        strongly_entangling_layers(qc, weights, wires=[0, 1], ranges=[1])
        names = [instr.operation.name for instr in qc.data]
        # phi,theta,omega -> rz, ry, rz per wire, then two CNOTs (range=1 on 2 wires: (0,1),(1,0))
        assert names == ["rz", "ry", "rz", "rz", "ry", "rz", "cx", "cx"]
        cx_pairs = [
            tuple(qc.find_bit(q).index for q in instr.qubits)
            for instr in qc.data
            if instr.operation.name == "cx"
        ]
        assert cx_pairs == [(0, 1), (1, 0)]

    def test_default_range_schedule(self):
        # r_l = (l mod (M-1)) + 1
        n_layers, n_wires = 3, 4
        weights = np.zeros((n_layers, n_wires, 3))
        qc = QuantumCircuit(n_wires)
        strongly_entangling_layers(qc, weights)
        # Full ring (n_wires CNOTs) applied once per layer, m > 1.
        cx_count = sum(1 for instr in qc.data if instr.operation.name == "cx")
        assert cx_count == n_wires * n_layers

    def test_rejects_bad_ranges(self):
        qc = QuantumCircuit(3)
        weights = np.zeros((1, 3, 3))
        with pytest.raises(ValidationError):
            strongly_entangling_layers(qc, weights, ranges=[3])  # 3 % 3 == 0, forbidden

    def test_rejects_shape_mismatch(self):
        qc = QuantumCircuit(3)
        with pytest.raises(ValidationError):
            strongly_entangling_layers(qc, np.zeros((1, 3, 2)))  # last dim must be 3


class TestAngleEmbedding:
    def test_matches_pennylane_docstring_example(self):
        # qp.AngleEmbedding.compute_decomposition([1., 2.], wires=["a","b"], rotation=qp.RX)
        # -> [RX(1., a), RX(2., b)]
        qc = QuantumCircuit(2)
        angle_embedding(qc, [1.0, 2.0], wires=[0, 1], rotation="X")
        assert [instr.operation.name for instr in qc.data] == ["rx", "rx"]
        assert qc.data[0].operation.params[0] == pytest.approx(1.0)
        assert qc.data[1].operation.params[0] == pytest.approx(2.0)

    def test_fewer_features_than_wires_leaves_remainder_untouched(self):
        qc = QuantumCircuit(3)
        angle_embedding(qc, [0.5], wires=[0, 1, 2], rotation="Y")
        assert len(qc.data) == 1

    def test_too_many_features_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValidationError):
            angle_embedding(qc, [0.1, 0.2, 0.3], wires=[0, 1])


class TestQAOAEmbedding:
    def test_shape_helper_matches_pennylane_formula(self):
        assert qaoa_embedding_shape(n_layers=2, n_wires=1) == (2, 1)
        assert qaoa_embedding_shape(n_layers=2, n_wires=2) == (2, 3)
        assert qaoa_embedding_shape(n_layers=2, n_wires=5) == (2, 10)

    def test_matches_pennylane_docstring_example_two_wires(self):
        # qp.QAOAEmbedding.compute_decomposition(features=[1.,2.], weights=[[0.1,-0.3,1.3],
        #   [0.9,-0.2,-2.1]], wires=["a","b"], local_field=qp.RY) ->
        # [RX(1,a), RX(2,b), MultiRZ(0.1,[a,b]), RY(-0.3,a), RY(1.3,b),
        #  RX(1,a), RX(2,b), MultiRZ(0.9,[a,b]), RY(-0.2,a), RY(-2.1,b),
        #  RX(1,a), RX(2,b)]
        qc = QuantumCircuit(2)
        features = [1.0, 2.0]
        weights = np.array([[0.1, -0.3, 1.3], [0.9, -0.2, -2.1]])
        qaoa_embedding(qc, features, weights, wires=[0, 1], local_field="Y")
        names = [instr.operation.name for instr in qc.data]
        assert names == [
            "rx", "rx", "rzz", "ry", "ry",
            "rx", "rx", "rzz", "ry", "ry",
            "rx", "rx",
        ]

    def test_qubit_and_param_count(self):
        n_wires, n_layers = 5, 3
        features = np.zeros(3)
        wire_dim = qaoa_embedding_shape(n_layers, n_wires)[1]
        weights = np.zeros((n_layers, wire_dim))
        qc = QuantumCircuit(n_wires)
        qaoa_embedding(qc, features, weights, wires=list(range(n_wires)))
        assert qc.num_qubits == n_wires
        # n_layers+1 encoding passes; RX for n_features, H for the rest, each pass
        rx_count = sum(1 for instr in qc.data if instr.operation.name == "rx")
        h_count = sum(1 for instr in qc.data if instr.operation.name == "h")
        assert rx_count == 3 * (n_layers + 1)
        assert h_count == (n_wires - 3) * (n_layers + 1)

    def test_too_wide_weights_raises(self):
        qc = QuantumCircuit(3)
        with pytest.raises(ValidationError):
            qaoa_embedding(qc, [0.1], np.zeros((1, 4)), wires=[0, 1, 2])  # expects width 6
