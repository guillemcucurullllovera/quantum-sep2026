"""Small seeded invariants on actual platform models; no empirical advantage claim."""
import json
import os

import numpy as np
from qiskit.quantum_info import Statevector

import algorithms
import applications
import core
import hardware
from algorithms.pqk.kernels import fidelity_gram
from algorithms.pqk.feature_maps import N_QUBITS
from algorithms.qae_iqae.barrier_circuits import build_A_barrier
from algorithms.qae_iqae.finance_integrands import build_crash_barrier
from algorithms.qaoa_qubo.gen2.mixers import initial_dicke_state, ring_xy_pairs
from algorithms.qaoa_qubo.gen2.qaoa_v2 import build_cost_hamiltonian, qaoa_statevector
from algorithms.qrc.reservoir import QuantumReservoir, QuantumReservoirConfig
from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from experiments.quantum1b.classical_exact import solve_exact_enumeration, solve_exact_milp
from experiments.quantum1b.generators import generate_instance


def run():
    if os.environ.get('ASSERT_INSTALLED_WHEEL')=='1':
        assert all('site-packages' in p.__file__ for p in (core, algorithms, applications, hardware))
    rng=np.random.default_rng(20261102)
    barrier=build_crash_barrier(crash_pct=.10,n=3)
    circuit=build_A_barrier(barrier.probs,barrier.barrier_mask)
    probability=Statevector.from_instruction(circuit).probabilities(qargs=[circuit.num_qubits-1])[1]
    np.testing.assert_allclose(probability,barrier.exact_grid_a,rtol=0,atol=1e-10)

    couplings=rng.normal(size=(4,4));couplings=(couplings+couplings.T)/2
    state=qaoa_statevector(initial_dicke_state(4,2),np.array([.2,.3]),np.array([.1,.4]),
                          build_cost_hamiltonian(couplings),ring_xy_pairs(4))
    np.testing.assert_allclose(np.vdot(state,state).real,1.,atol=1e-10)
    outside=[i for i in range(16) if i.bit_count()!=2]
    assert np.abs(state[outside]).max()<1e-10

    gram=fidelity_gram(rng.uniform(0,1,size=(5,N_QUBITS)),'AngleCNOT',psd_correct=False)
    np.testing.assert_allclose(np.diag(gram),1.,atol=1e-10)
    assert np.linalg.eigvalsh(gram).min()>-1e-9

    reservoir=QuantumReservoir(QuantumReservoirConfig(n1=2,n2=2,seed=61)).build()
    features=reservoir.sequence_features(rng.uniform(-np.pi,np.pi,size=(12,2)),washout=3)
    assert features.shape==(9,4) and np.isfinite(features).all()
    versions=[]
    for version in VQCVersion:
        ansatz=VQCAnsatz(n_qubits=4,version=version)
        psi=ansatz.statevector(rng.uniform(-np.pi,np.pi,4),ansatz.init_params(seed=61))
        np.testing.assert_allclose(np.vdot(psi,psi).real,1.,atol=1e-9)
        versions.append(version.value)
    for family in ('independent','correlated','tight_budget'):
        instance=generate_instance(family,6,seed=61)
        enumeration=solve_exact_enumeration(instance)
        milp=solve_exact_milp(instance)
        assert enumeration.feasible and milp.feasible
        np.testing.assert_allclose(enumeration.objective,milp.objective,rtol=0,atol=1e-6)
    return {'data':'seeded synthetic only; not historical performance','seed':20261102,
            'qae_barrier_grid_oracle':'PASS','qaoa_norm_and_cardinality':'PASS',
            'fidelity_kernel_psd':'PASS','qrc_finite_readout':'PASS','vqc_versions_normalized':versions,
            'milp_enumeration_crosschecks':3}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
