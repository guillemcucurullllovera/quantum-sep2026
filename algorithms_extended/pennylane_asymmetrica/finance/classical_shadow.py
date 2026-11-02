"""Classical shadow tomography, adapted from PennyLane's
``pennylane.shadows.classical_shadow`` (Apache-2.0, Xanadu Quantum
Technologies Inc.) — see ``upstream/shadows/classical_shadow.py`` for the
original and ``THIRD_PARTY_NOTICES.md`` for what changed.

Reimplemented against Qiskit (random-Pauli-basis circuit measurement via
Aer, not PennyLane's device/tape machinery). The core estimator math
(median-of-means over random-Pauli-basis snapshots) is unchanged from
upstream, per Huang, Kueng & Preskill 2020 (arXiv:2002.08953).

This is a MEASUREMENT technique -- reconstructing expectation values of
many observables from few, cheap, randomized measurements of one quantum
state. It is unrelated to ``hardware/shadow/`` (shadow-deployment
module, which freezes a *trained model's weights* for low-latency
production inference). Both are legitimately named "shadow" for different
reasons; do not conflate them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

_PAULI_RECIPE = {0: "X", 1: "Y", 2: "Z"}


@dataclass
class ClassicalShadow:
    """A collection of randomized-basis measurement snapshots of one
    quantum state, from which many Pauli-observable expectation values can
    be estimated without re-measuring for each one.
    """

    bits: np.ndarray  # shape (T, n), 0/1 outcomes
    recipes: np.ndarray  # shape (T, n), 0=X, 1=Y, 2=Z basis used per snapshot

    @property
    def n_snapshots(self) -> int:
        return self.bits.shape[0]

    @property
    def n_qubits(self) -> int:
        return self.bits.shape[1]

    def expval(self, word: np.ndarray, num_batches: int = 10) -> float:
        """Estimate <psi| P |psi> for a Pauli word (array of 0=X,1=Y,2=Z,-1=I
        per qubit) via median-of-means over the recorded snapshots."""
        per_snapshot = pauli_expval(self.bits, self.recipes, word)
        return float(median_of_means(per_snapshot, num_batches))


def sample_classical_shadow(circuit: QuantumCircuit, n_snapshots: int, seed: int | None = None) -> ClassicalShadow:
    """Run ``circuit`` (state-preparation only, no measurements) ``n_snapshots``
    times, each time in a freshly-random single-qubit Pauli basis, and record
    the resulting bitstrings + basis choices. This is the randomized-basis
    measurement protocol classical shadows are built from.
    """
    rng = np.random.default_rng(seed)
    n = circuit.num_qubits
    backend = AerSimulator()

    recipes = rng.integers(0, 3, size=(n_snapshots, n))
    bits = np.zeros((n_snapshots, n), dtype=int)

    # group snapshots by identical recipe to batch simulator calls (recipes
    # are usually all-distinct for reasonable n_snapshots/n, so this mostly
    # runs one-at-a-time -- still correct either way, just not maximally fast)
    for t in range(n_snapshots):
        qc = circuit.copy()
        for q in range(n):
            basis = _PAULI_RECIPE[recipes[t, q]]
            if basis == "X":
                qc.h(q)
            elif basis == "Y":
                qc.sdg(q)
                qc.h(q)
            # Z: measure in computational basis directly, no rotation
        qc.measure_all()
        result = backend.run(qc, shots=1, seed_simulator=int(rng.integers(0, 2**31))).result()
        counts = result.get_counts()
        outcome = next(iter(counts))  # single shot -> one bitstring
        # Qiskit bitstrings are little-endian (qubit 0 rightmost) with a
        # space before the classical-register-0 measure_all() adds; strip it
        outcome = outcome.replace(" ", "")
        bits[t] = [int(b) for b in reversed(outcome)][:n]

    return ClassicalShadow(bits=bits, recipes=recipes)


def median_of_means(arr: np.ndarray, num_batches: int) -> float:
    """Median of batch-means; adapted from
    ``pennylane.shadows.classical_shadow.median_of_means``. Robust to the
    heavy-tailed per-snapshot estimator variance classical shadows produce."""
    arr = np.asarray(arr, dtype=float)
    batch_size = int(np.ceil(arr.shape[0] / num_batches))
    means = [arr[i * batch_size:(i + 1) * batch_size].mean() for i in range(num_batches) if arr[i * batch_size:(i + 1) * batch_size].size > 0]
    return float(np.median(means))


def pauli_expval(bits: np.ndarray, recipes: np.ndarray, word: np.ndarray) -> np.ndarray:
    """Per-snapshot Pauli-word expectation-value estimator; adapted verbatim
    (math unchanged) from ``pennylane.shadows.classical_shadow.pauli_expval``.

    Snapshots whose recorded basis doesn't match ``word`` on every non-identity
    qubit contribute 0 (they carry no information about this particular word);
    matching snapshots contribute +/-1 * 3^k where k = number of non-identity
    qubits in the word (the classical-shadow inverse-channel normalization).
    """
    bits, recipes, word = np.asarray(bits), np.asarray(recipes), np.asarray(word)
    T, n = bits.shape
    non_identity = word != -1
    k = int(non_identity.sum())
    match = np.all((recipes == word) | (~non_identity)[None, :], axis=1)
    parity = (bits[:, non_identity].sum(axis=1)) % 2
    values = np.where(match, (1 - 2 * parity) * (3**k), 0.0)
    return values.astype(float)
