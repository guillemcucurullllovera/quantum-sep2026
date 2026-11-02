"""IQAE amplitude estimation for a factored barrier/comparator circuit — canonical
`qiskit_algorithms.IterativeAmplitudeEstimation` (Grinko et al., Apache-2.0), rigorous
Clopper-Pearson CI. Runs ideal + FakeKingston-noisy, plus a counts-level M3
readout-mitigation check. No advantage claim (see THIRD_PARTY_NOTICES.md).

Honest caveat: IQAE's confidence interval is a *statistical* guarantee under a
noiseless-oracle assumption; under real device noise the estimate is biased, and the
bias can fall OUTSIDE the (tight) CI — the CI bounds shot noise, not systematic device
error. Every result surfaces this in its `provenance.note` field, not only in prose.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Optional

from algorithms.qae_iqae.finance_integrands import BarrierIntegrand
from algorithms.qae_iqae.barrier_circuits import build_A_barrier, comparator_affinity
from core.config import get_settings
from core.errors import CredentialError

_BASIS = ["ry", "rz", "rx", "cx", "x", "h", "sx", "cz", "id"]


def _decompose(A):
    """Pre-decompose to plain gates — a StatePreparation composite under Grover can
    segfault Aer if left un-decomposed."""
    from qiskit import transpile
    return transpile(A, basis_gates=_BASIS, optimization_level=1)


def _versions() -> Dict[str, str]:
    import qiskit, qiskit_aer, qiskit_algorithms, mthree
    return {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
            "qiskit_algorithms": qiskit_algorithms.__version__, "mthree": mthree.__version__}


def run_iqae(A, objective_qubit: int, *, sampler, transpiler, epsilon: float = 0.01,
             alpha: float = 0.05) -> Dict:
    from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
    Ad = _decompose(A)
    problem = EstimationProblem(state_preparation=Ad, objective_qubits=[objective_qubit])
    iqae = IterativeAmplitudeEstimation(epsilon, alpha, confint_method="beta",
                                        sampler=sampler, transpiler=transpiler)
    res = iqae.estimate(problem)
    lo, hi = float(res.confidence_interval[0]), float(res.confidence_interval[1])
    return {"a_hat": float(res.estimation), "ci": [lo, hi], "ci_halfwidth": (hi - lo) / 2.0,
            "oracle_queries": int(res.num_oracle_queries), "epsilon": epsilon, "alpha": alpha}


def _grover_metadata(A, anc: int, fake, ks=(0, 1, 2)) -> Dict:
    from qiskit import transpile
    from qiskit_algorithms import EstimationProblem
    Ad = _decompose(A)
    problem = EstimationProblem(state_preparation=Ad, objective_qubits=[anc])
    Q = problem.grover_operator
    out = {}
    for k in ks:
        circ = Ad.copy()
        for _ in range(k):
            circ.compose(Q, inplace=True)
        circ.measure_all()
        t = transpile(circ, backend=fake, optimization_level=0, seed_transpiler=7)
        gc = {kk: int(vv) for kk, vv in t.count_ops().items()}
        twoq = sum(v for g, v in gc.items() if g in ("cx", "cz", "ecr", "rzz"))
        out[f"Q{k}A"] = {"depth": int(t.depth()), "twoq_gates": twoq, "total_ops": sum(gc.values())}
    return out


def _m3_check(A, anc: int, fake, shots: int) -> Dict:
    from qiskit import transpile
    from mthree.utils import final_measurement_mapping
    from algorithms.qae_iqae import mitigation as MIT

    Am = A.copy()
    Am.measure_all()
    t = transpile(Am, backend=fake, optimization_level=0, seed_transpiler=7)
    mapping = final_measurement_mapping(t)
    counts = fake.run(t, shots=shots).result().get_counts()
    r = MIT.apply_m3(fake, counts, mapping, shots, ancilla_clbit=anc)
    return {"p1_raw": r.p1_raw, "p1_mitigated": r.p1_mitigated,
            "avg_readout_fidelity": r.avg_readout_fidelity,
            "readout_fidelities": list(r.readout_fidelities)}


def run_barrier_demo(barrier: BarrierIntegrand, *, epsilon: float = 0.01, alpha: float = 0.05,
                     shots: int = 2048, seed: int = 11, m3: bool = True,
                     timestamp: Optional[str] = None) -> Dict:
    """The platform's primary simulator-only QAE demo path: ideal + Kingston-noisy IQAE
    on the crash-barrier / return-distribution comparator, with an M3 mitigation check.
    Requires no credentials."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    n = int(round(math.log2(len(barrier.probs))))
    anc = n
    A = build_A_barrier(barrier.probs, barrier.barrier_mask)
    fake = FakeKingston()

    out: Dict = {
        "integrand": {"name": barrier.name, "label": barrier.label, "kind": barrier.kind,
                      "meta": dict(barrier.meta)},
        "estimand": "P(S_T <= barrier) or P(PnL <= threshold) — risk-neutral/empirical CDF "
                    "(the Woerner-Egger VaR building block); pure comparator, no payoff rescaling.",
        "estimator": "IQAE (qiskit-algorithms IterativeAmplitudeEstimation, Clopper-Pearson CI)",
        "exact_grid_a": barrier.exact_grid_a, "exact_analytic": barrier.exact_analytic,
        "discretisation_gap": barrier.meta.get("discretisation_gap"),
        "config": {"n_index_qubits": n, "n_qubits": n + 1, "epsilon": epsilon, "alpha": alpha,
                   "shots": shots, "backend": fake.name, "backend_num_qubits": int(fake.num_qubits),
                   "opt_level_noisy": 0},
        "modes": {}, "circuits": {}, "comparator": {}, "m3": {},
        "provenance": {
            "versions": _versions(),
            "timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat(),
            "note": f"n={n} ({2**n}-point) DEMO — coarse, not an accurate probability, no advantage. "
                    "IQAE CI is a statistical (shot-noise) guarantee under a noiseless-oracle assumption.",
        },
    }

    ideal_pm = generate_preset_pass_manager(1, AerSimulator())
    ideal = run_iqae(A, anc, sampler=SamplerV2(default_shots=shots, seed=seed),
                     transpiler=ideal_pm, epsilon=epsilon, alpha=alpha)
    ideal["abs_error_vs_grid"] = abs(ideal["a_hat"] - barrier.exact_grid_a)
    out["modes"]["ideal"] = ideal

    noise = NoiseModel.from_backend(fake)
    noisy_sampler = SamplerV2(default_shots=shots, seed=seed,
                              options={"backend_options": {"noise_model": noise}})
    noisy_pm = generate_preset_pass_manager(0, fake)
    noisy = run_iqae(A, anc, sampler=noisy_sampler, transpiler=noisy_pm, epsilon=epsilon, alpha=alpha)
    noisy["abs_error_vs_grid"] = abs(noisy["a_hat"] - barrier.exact_grid_a)
    noisy["true_grid_a_inside_ci"] = noisy["ci"][0] <= barrier.exact_grid_a <= noisy["ci"][1]
    out["modes"]["kingston_noisy"] = noisy

    out["circuits"] = _grover_metadata(A, anc, fake, ks=(0, 1, 2))
    resid, is_affine = comparator_affinity(barrier.barrier_mask)
    out["comparator"] = {"affine_residual": resid, "is_affine": is_affine,
                         "note": "comparator only; the distribution LOADER dominates circuit depth"}
    if m3:
        out["m3"] = _m3_check(A, anc, fake, shots)
    return out


def run_on_hardware(barrier: BarrierIntegrand, *, backend_name: str = "ibm_kingston",
                    epsilon: float = 0.03, alpha: float = 0.05, shots: int = 2048) -> Optional[Dict]:
    """Real IBM Quantum run via IQAE (Qiskit Runtime). Falls back to a token-free DRY RUN
    (transpile against FakeKingston + report metadata) when no credential is present.
    Never fabricates a result. Credential comes ONLY from core.config — never a
    parameter, never hardcoded, never logged."""
    settings = get_settings()
    n = int(round(math.log2(len(barrier.probs))))
    anc = n
    A = build_A_barrier(barrier.probs, barrier.barrier_mask)

    if not settings.ibm.is_present:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        fake = FakeKingston()
        meta = _grover_metadata(A, anc, fake, ks=(0, 1, 2))
        return {
            "mode": "dry_run", "reason": "IBM_QUANTUM_TOKEN not set",
            "isa_target": fake.name, "backend_num_qubits": int(fake.num_qubits),
            "exact_grid_a": barrier.exact_grid_a, "exact_analytic": barrier.exact_analytic,
            "circuits": meta,
            "note": "Submission requires IBM_QUANTUM_TOKEN in the environment. Prefer a looser "
                    "epsilon on real hardware — tight-epsilon deep-Grover circuits are noise-dominated.",
        }

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    token = settings.ibm.require().token  # raises CredentialError if somehow absent here
    service = QiskitRuntimeService(channel=settings.ibm.channel, token=token)
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(0, backend)
    res = run_iqae(A, anc, sampler=SamplerV2(mode=backend), transpiler=pm, epsilon=epsilon, alpha=alpha)
    res["exact_grid_a"] = barrier.exact_grid_a
    res["abs_error_vs_grid"] = abs(res["a_hat"] - barrier.exact_grid_a)
    res["backend"] = backend.name
    res["mode"] = "hardware"
    res["note"] = "REAL IBM Quantum IQAE run — no advantage claim."
    return res
