"""Tests for `experiments.quantum1b.hardware_readiness.lineage`.

Uses FakeKingston (a real device-calibration snapshot shipped with
qiskit-ibm-runtime, no credentials/network required) -- the same backend
`algorithms/qae_iqae/backends.py` already uses elsewhere in this estate.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.schemas import BackendDescriptor, BackendKind, HardwareRunEvidence
from experiments.quantum1b.generators import generate_instance
from experiments.quantum1b.qubo_encoding import build_qubo
from experiments.quantum1b.hardware_readiness.lineage import (
    build_lineage_record,
    build_qaoa_circuit_for_lineage,
    capture_transpilation_metadata,
)


@pytest.fixture(scope="module")
def fake_backend():
    from algorithms.qae_iqae.backends import kingston_backend

    return kingston_backend()


class TestBuildQaoaCircuitForLineage:
    def test_builds_a_real_circuit_matching_qubit_count(self):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        n = qubo.n_qubits()
        rng = np.random.default_rng(0)
        gamma, beta = rng.uniform(0, np.pi, 1), rng.uniform(0, np.pi, 1)

        qc = build_qaoa_circuit_for_lineage(qubo.Q, p=1, gamma=gamma, beta=beta)

        assert qc.num_qubits == n
        assert qc.num_clbits == n


class TestCaptureTranspilationMetadata:
    def test_captures_real_metadata_against_fake_backend(self, fake_backend):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        rng = np.random.default_rng(0)
        gamma, beta = rng.uniform(0, np.pi, 1), rng.uniform(0, np.pi, 1)
        qc = build_qaoa_circuit_for_lineage(qubo.Q, p=1, gamma=gamma, beta=beta)

        transpiled = capture_transpilation_metadata(qc, fake_backend, seed=0)

        assert transpiled.depth > 0
        assert sum(transpiled.gate_counts.values()) > 0
        assert transpiled.twoq_gates >= 0
        # a transpiled circuit against a real coupling map is generally at
        # least as deep as the logical circuit once SWAP routing is
        # accounted for -- just assert it's a sane positive integer, not a
        # specific value (that would be brittle against transpiler version
        # changes).
        assert isinstance(transpiled.depth, int)


class TestRunLineageRecord:
    def test_record_hash_is_deterministic(self, fake_backend):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        rng = np.random.default_rng(0)
        gamma, beta = rng.uniform(0, np.pi, 1), rng.uniform(0, np.pi, 1)
        qc = build_qaoa_circuit_for_lineage(qubo.Q, p=1, gamma=gamma, beta=beta)
        transpiled = capture_transpilation_metadata(qc, fake_backend, seed=0)

        backend_desc = BackendDescriptor(name="FakeKingston", kind=BackendKind.FAKE, num_qubits=fake_backend.num_qubits)

        r1 = build_lineage_record(
            instance_fingerprint=inst.fingerprint(),
            transpiled=transpiled,
            qubo_n_qubits=qubo.n_qubits(),
            backend=backend_desc,
            shots=4096,
            seed=0,
        )
        r2 = build_lineage_record(
            instance_fingerprint=inst.fingerprint(),
            transpiled=transpiled,
            qubo_n_qubits=qubo.n_qubits(),
            backend=backend_desc,
            shots=4096,
            seed=0,
        )
        assert r1.record_hash() == r2.record_hash()

    def test_record_hash_changes_with_different_backend(self, fake_backend):
        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        rng = np.random.default_rng(0)
        gamma, beta = rng.uniform(0, np.pi, 1), rng.uniform(0, np.pi, 1)
        qc = build_qaoa_circuit_for_lineage(qubo.Q, p=1, gamma=gamma, beta=beta)
        transpiled = capture_transpilation_metadata(qc, fake_backend, seed=0)

        r_fake = build_lineage_record(
            instance_fingerprint=inst.fingerprint(),
            transpiled=transpiled,
            qubo_n_qubits=qubo.n_qubits(),
            backend=BackendDescriptor(name="FakeKingston", kind=BackendKind.FAKE, num_qubits=fake_backend.num_qubits),
            shots=4096,
            seed=0,
        )
        r_sim = build_lineage_record(
            instance_fingerprint=inst.fingerprint(),
            transpiled=transpiled,
            qubo_n_qubits=qubo.n_qubits(),
            backend=BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=32),
            shots=4096,
            seed=0,
        )
        assert r_fake.record_hash() != r_sim.record_hash()

    def test_to_json_dict_is_json_serializable(self, fake_backend):
        import json

        inst = generate_instance("independent", 4, seed=0)
        qubo = build_qubo(inst)
        rng = np.random.default_rng(0)
        gamma, beta = rng.uniform(0, np.pi, 1), rng.uniform(0, np.pi, 1)
        qc = build_qaoa_circuit_for_lineage(qubo.Q, p=1, gamma=gamma, beta=beta)
        transpiled = capture_transpilation_metadata(qc, fake_backend, seed=0)

        evidence = HardwareRunEvidence(
            backend_name="FakeKingston",
            job_id="job-lineage-test",
            execution_timestamp="2026-09-30T12:00:00+00:00",
        )
        record = build_lineage_record(
            instance_fingerprint=inst.fingerprint(),
            transpiled=transpiled,
            qubo_n_qubits=qubo.n_qubits(),
            backend=BackendDescriptor(name="FakeKingston", kind=BackendKind.FAKE, num_qubits=fake_backend.num_qubits),
            shots=4096,
            seed=0,
            hardware_evidence=evidence,
        )
        d = record.to_json_dict()
        json.dumps(d)  # must not raise
        assert d["hardware_evidence"]["job_id"] == "job-lineage-test"
        assert "record_hash" in d
