"""IBM Kingston (Heron r3) hardware runner for the QAE pipeline — Qiskit Runtime SamplerV2 + M3.

HISTORICAL PROTOTYPE, preserved as-is. This is the platform's real-QPU
execution path for QAE, preserved as-is: **it is blocked by design until an IBM Quantum
token is available** (dry-run transpiles against a fake backend and validates on Aer; it
never fabricates a hardware result). Credential handling here reads only
`IBM_QUANTUM_TOKEN`/`QISKIT_IBM_TOKEN` from the environment or an explicit `--token` CLI
argument at invocation time — no credential is stored, logged, or hardcoded anywhere in
this file. Production QAE work's credential path goes through `core.config` instead
(see `algorithms.qae_iqae.iqae`); this CLI is kept close to its original form as real
development history, not modernized.

  # validate without a token (transpiles against FakeKingston, checks on Aer):
  python -m prototypes.qae.run_ibm --integrand g2 --ks 0 1 2 --dry-run
  # real run (needs a token in IBM_QUANTUM_TOKEN / QISKIT_IBM_TOKEN, or --token):
  python -m prototypes.qae.run_ibm --integrand finance_call_spread --ks 0 1 2 --shots 4096 --wait

Honesty rails: n=2 pipeline demonstration, no advantage claim.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from algorithms_extended.qae_spinqit import mlae as M
from algorithms_extended.qae_spinqit.state_prep import angle_table, build_qae_circuit, exact_amplitude, exact_pk
from prototypes.qae._compat import get_integrand


def _token(explicit: Optional[str]) -> Optional[str]:
    return explicit or os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")


def select_qubits(backend, n: int = 3) -> List[int]:
    """Pick `n` connected physical qubits (a path) with the lowest total readout error, so the
    ancilla (returned last, as the path centre) is adjacent to both index qubits."""
    target = backend.target
    num = backend.num_qubits

    def ro_err(q: int) -> float:
        try:
            return float(target["measure"][(q,)].error)
        except Exception:
            return 1.0

    try:
        cmap = list(backend.coupling_map)
    except Exception:
        cmap = list(target.build_coupling_map())
    neigh: Dict[int, set] = {q: set() for q in range(num)}
    for a, b in cmap:
        neigh[a].add(b)
        neigh[b].add(a)

    best: Optional[Tuple[float, List[int]]] = None
    for c in range(num):
        ns = sorted(neigh[c], key=ro_err)
        if len(ns) >= 2:
            n1, n2 = ns[0], ns[1]
            score = ro_err(n1) + ro_err(n2) + ro_err(c)
            if best is None or score < best[0]:
                best = (score, [n1, n2, c])  # index0, index1, ancilla(centre)
    if best is None:
        return list(range(n))
    return best[1]


def _aer_check(at, ks: Sequence[int], shots: int = 65536) -> List[Dict[str, float]]:
    """Ideal-Aer sanity check per k (no hardware)."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    from algorithms.qae_iqae.backends import ancilla_p1
    sim = AerSimulator()
    rows = []
    for k in ks:
        qc = transpile(build_qae_circuit(at, k, measure=True), sim)
        counts = sim.run(qc, shots=shots).result().get_counts()
        s, ntot = ancilla_p1(counts)
        rows.append({"k": int(k), "p_hat": s / ntot, "p_model": exact_pk(at, k)})
    return rows


def main():
    p = argparse.ArgumentParser(description="Run the QAE pipeline on IBM Kingston (Heron r3).")
    p.add_argument("--integrand", default="finance_call_spread")
    p.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--rule", default="midpoint", choices=["left", "midpoint", "right"])
    p.add_argument("--backend", default="ibm_kingston")
    p.add_argument("--token", default=None)
    p.add_argument("--instance", default=None, help="IBM Cloud instance CRN (if required)")
    p.add_argument("--dry-run", action="store_true", help="transpile + Aer check only, no submission")
    p.add_argument("--wait", action="store_true", help="block until jobs finish and postprocess")
    p.add_argument("--no-m3", action="store_true", help="skip M3 readout mitigation")
    p.add_argument("--out", default="prototypes/qae/results")
    args = p.parse_args()

    ig = get_integrand(args.integrand)
    at = angle_table(ig, rule=args.rule)
    exact_a = exact_amplitude(at)
    ks = [int(k) for k in args.ks]

    print(f"integrand={ig.name} ({ig.label})  exact_a={exact_a:.5f}  ks={ks}  shots={args.shots}")
    print("Aer ideal check (p_hat vs model):")
    for row in _aer_check(at, ks):
        print(f"  k={row['k']}: p_hat={row['p_hat']:.4f}  model={row['p_model']:.4f}")

    # ---- DRY RUN: transpile against FakeKingston (stand-in ISA), no submission ----
    if args.dry_run or _token(args.token) is None:
        from algorithms.qae_iqae.backends import kingston_backend, transpile_for
        fake = kingston_backend()
        layout = select_qubits(fake, 3)
        print(f"\n[dry-run] target ISA = {fake.name} ({fake.num_qubits}q); "
              f"selected layout (q0,q1,anc)={layout}")
        for k in ks:
            tc = transpile_for(build_qae_circuit(at, k, measure=True), fake,
                               optimization_level=0, initial_layout=layout)
            print(f"  k={k}: depth={tc.depth}  2q={tc.twoq_gates}  ops={tc.gate_counts}")
        if _token(args.token) is None and not args.dry_run:
            print("\n[BLOCKED] No IBM token (IBM_QUANTUM_TOKEN/QISKIT_IBM_TOKEN unset). "
                  "Submission deferred. Circuits are submission-ready.")
        else:
            print("\n[dry-run OK] circuits transpile cleanly; re-run with a token (no --dry-run) to submit.")
        return

    # ---- REAL SUBMISSION ----
    token = _token(args.token)
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from algorithms.qae_iqae.backends import ancilla_p1
    from algorithms.qae_iqae import mitigation as MIT

    kwargs = {"channel": "ibm_quantum_platform", "token": token}
    if args.instance:
        kwargs["instance"] = args.instance
    service = QiskitRuntimeService(**kwargs)
    backend = service.backend(args.backend)
    layout = select_qubits(backend, 3)
    print(f"\nBackend {backend.name}: selected layout (q0,q1,anc)={layout}")

    transpiled = {k: transpile(build_qae_circuit(at, k, measure=True), backend=backend,
                               optimization_level=0, initial_layout=layout) for k in ks}
    sampler = Sampler(backend)
    jobs = {k: sampler.run([transpiled[k]], shots=args.shots) for k in ks}
    for k, j in jobs.items():
        print(f"  submitted k={k}: job_id={j.job_id()}")
    if not args.wait:
        print("Jobs submitted (no --wait). Retrieve later and re-run analysis.")
        return

    mit = None
    if not args.no_m3:
        from mthree.utils import final_measurement_mapping
        m = final_measurement_mapping(transpiled[ks[0]])
        mit = MIT.build_mitigator(backend, m)

    succ_raw, succ_m3, shot_list = [], [], []
    m3_meta = {}
    for k in ks:
        res = jobs[k].result()
        counts = res[0].data.c.get_counts()
        s, ntot = ancilla_p1(counts)
        succ_raw.append(s); shot_list.append(ntot)
        if mit is not None:
            from mthree.utils import final_measurement_mapping
            mp = final_measurement_mapping(transpiled[k])
            r = MIT.apply_m3(backend, counts, mp, ntot, mitigator=mit)
            succ_m3.append(r.successes_mitigated)
            m3_meta = {"avg_readout_fidelity": r.avg_readout_fidelity,
                       "readout_fidelities": list(r.readout_fidelities)}

    raw = M.estimate(ks, succ_raw, shot_list)
    out = {
        "integrand": {"name": ig.name, "label": ig.label, "meta": dict(ig.meta)},
        "backend": backend.name, "layout": layout, "ks": ks, "shots": args.shots,
        "exact_amplitude": exact_a,
        "raw": M.MLAEResult.as_dict(raw), "abs_error_raw": abs(raw.a_hat - exact_a),
        "provenance": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "note": "REAL IBM Kingston hardware run — n=2 pipeline demo, no advantage claim."},
    }
    print(f"\nRAW  a_hat={raw.a_hat:.5f}  err={abs(raw.a_hat-exact_a):.5f}  CI={raw.ci_bootstrap}")
    if succ_m3:
        m3 = M.estimate(ks, succ_m3, shot_list)
        out["m3"] = M.MLAEResult.as_dict(m3)
        out["abs_error_m3"] = abs(m3.a_hat - exact_a)
        out["m3_meta"] = m3_meta
        print(f"+M3  a_hat={m3.a_hat:.5f}  err={abs(m3.a_hat-exact_a):.5f}  CI={m3.ci_bootstrap}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"qae_HARDWARE_{ig.name}_{backend.name}_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
