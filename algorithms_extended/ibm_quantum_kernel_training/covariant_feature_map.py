# Adapted from qiskit-community/prototype-quantum-kernel-training,
# qkt/feature_maps/covariant_feature_map.py
#
# (C) Copyright IBM 2021.
#
# The upstream file this is adapted from is licensed under the Apache
# License, Version 2.0. You may obtain a copy of this license at
# http://www.apache.org/licenses/LICENSE-2.0. Per the upstream file's own
# header, any modifications or derivative works of that code must retain
# its copyright notice and carry a notice indicating alteration -- this
# file is such a notice: the circuit *architecture* (trainable per-qubit
# Ry pre-rotation, followed by a group/entanglement layer, followed by a
# data-encoding Rz/Rx layer with two classical features per qubit) is
# adapted from the upstream design; the implementation below is written
# independently against this platform's Qiskit version and conventions
# (not copied line-for-line), and SectorCovariantFeatureMap below is new
# Asymmetrica-original work with no upstream equivalent.
"""Covariant quantum feature maps: kernels aware of a known group structure
in the data, per Glick et al. 2024 ("Covariant quantum kernels for data with
group structure", arXiv:2105.03406), the paper this upstream toolkit was
built to accompany.

Where `algorithms.pqk.feature_maps` module treats every feature
pair symmetrically (fixed linear/ring entanglement), a covariant map's
entanglement pattern is chosen to respect a *known* symmetry in the data --
here, finance's natural analogue is sector/asset-class structure: assets in
the same sector should be treated with more symmetry (denser, permutation-
covariant entanglement) than assets in different sectors. `CovariantFeatureMap`
below is the general (ungrouped) upstream architecture; `SectorGroupFeatureMap`
is the finance-specific extension.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class CovariantFeatureMap(QuantumCircuit):
    """General covariant feature map: trainable Ry layer, then a fixed
    entanglement pattern (the "group" structure), then a data-encoding
    Rz/Rx layer with 2 classical features per qubit.

    Architecture matches the upstream toolkit's `CovariantFeatureMap`;
    reimplemented independently (not copy-pasted) against this platform's
    Qiskit version.
    """

    def __init__(
        self,
        feature_dimension: int,
        entanglement: Sequence[tuple[int, int]] | Callable[[int], list[tuple[int, int]]] | None = None,
        single_training_parameter: bool = False,
        name: str = "CovariantFeatureMap",
    ) -> None:
        if feature_dimension % 2 != 0:
            raise ValueError("Covariant feature map requires an even number of input features.")
        self.feature_dimension = feature_dimension
        self.single_training_parameter = single_training_parameter
        num_qubits = feature_dimension // 2
        super().__init__(num_qubits, name=name)

        self.entanglement = (
            entanglement if entanglement is not None else [(i, i + 1) for i in range(num_qubits - 1)]
        )
        if callable(self.entanglement):
            self.entanglement = self.entanglement(num_qubits)

        self.input_parameters = ParameterVector("x_par", feature_dimension)
        n_train = 1 if single_training_parameter else num_qubits
        self.training_parameters = ParameterVector("theta_par", n_train)

        for i in range(num_qubits):
            theta = self.training_parameters[0] if single_training_parameter else self.training_parameters[i]
            self.ry(theta, self.qubits[i])
        for source, target in self.entanglement:
            self.cz(self.qubits[source], self.qubits[target])
        self.barrier()
        for i in range(num_qubits):
            self.rz(-2 * self.input_parameters[2 * i + 1], self.qubits[i])
            self.rx(-2 * self.input_parameters[2 * i], self.qubits[i])

    def bind_data(self, x: np.ndarray) -> QuantumCircuit:
        return self.assign_parameters(dict(zip(self.input_parameters, x)), inplace=False)

    def bind_training(self, theta: np.ndarray) -> QuantumCircuit:
        return self.assign_parameters(dict(zip(self.training_parameters, theta)), inplace=False)


def sector_group_entanglement(sector_of_qubit: Sequence[int]) -> list[tuple[int, int]]:
    """Build a group-covariant entanglement pattern for finance data with a
    known sector/asset-class structure: fully connect qubits within the same
    sector (dense, permutation-covariant treatment of same-sector assets),
    and connect one representative pair between adjacent sectors (sparse
    cross-sector coupling, avoiding O(n^2) circuit depth). New Asymmetrica
    work, no upstream equivalent -- upstream's toolkit only demonstrates a
    fixed abstract group structure, not a concrete finance grouping.
    """
    n = len(sector_of_qubit)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sector_of_qubit[i] == sector_of_qubit[j]:
                edges.append((i, j))
    sectors_seen: dict[int, int] = {}
    for idx, s in enumerate(sector_of_qubit):
        if s not in sectors_seen:
            sectors_seen[s] = idx
    reps = list(sectors_seen.values())
    for a, b in zip(reps, reps[1:]):
        if (a, b) not in edges and (b, a) not in edges:
            edges.append((a, b))
    return edges


class SectorGroupFeatureMap(CovariantFeatureMap):
    """Covariant feature map with entanglement wired to a financial sector
    grouping. Each pair of classical features (e.g. return, volatility) maps
    to one qubit representing one asset; `sector_of_qubit[i]` names the
    sector of the asset on qubit i.
    """

    def __init__(self, sector_of_qubit: Sequence[int], single_training_parameter: bool = False) -> None:
        self.sector_of_qubit = list(sector_of_qubit)
        n_qubits = len(sector_of_qubit)
        super().__init__(
            feature_dimension=2 * n_qubits,
            entanglement=sector_group_entanglement(self.sector_of_qubit),
            single_training_parameter=single_training_parameter,
            name="SectorGroupFeatureMap",
        )
