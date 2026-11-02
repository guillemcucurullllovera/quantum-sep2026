"""Run a QAE campaign on the simulators (ideal + IBM Kingston noise model) and emit a JSON artifact.

HISTORICAL PROTOTYPE, preserved as-is. Production QAE work now lives in
`algorithms.qae_iqae` (credential handling via `core.config`, results via
`core.experiment_registry`); this script remains the original, still-useful
campaign-runner CLI.

Modes:
  * ideal          — noiseless statevector-accurate sampling.
  * kingston_noisy  — FakeKingston (real Heron-r3 calibration) noise model, opt_level=0 transpile.
  * kingston_m3     — kingston_noisy + M3 readout mitigation.

Usage:
  python -m prototypes.qae.run_sim --integrand finance_call_spread --ks 0 1 2 --shots 4096 --out <dir>
  python -m prototypes.qae.run_sim --integrand g2 --ks 0 1 2 --shots 4096

Honesty rails: at n=2 this is a *pipeline validation*, not a price or an advantage.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from algorithms.qae_iqae import backends as B
from algorithms.qae_iqae import mitigation as MIT
from algorithms_extended.qae_spinqit import mlae as M
from algorithms_extended.qae_spinqit.state_prep import angle_table, build_qae_circuit, exact_amplitude, exact_pk
from algorithms_extended.qae_spinqit.toy_integrands import Integrand
from prototypes.qae._compat import get_integrand


def _versions() -> Dict[str, str]:
    import qiskit, qiskit_aer, qiskit_ibm_runtime, mthree
    return {
        "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
        "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__, "mthree": mthree.__version__,
    }


def _mlae_block(res: M.MLAEResult) -> Dict[str, object]:
    return {
        "a_hat": res.a_hat, "theta_hat": res.theta_hat, "p_hat": list(res.p_hat),
        "successes": list(res.successes), "shots": list(res.shots),
        "se_fisher": res.se_fisher, "ci_fisher": list(res.ci_fisher),
        "ci_bootstrap": list(res.ci_bootstrap), "nll": res.nll,
    }


def run_campaign(
    integrand: Integrand,
    ks: Sequence[int] = (0, 1, 2),
    shots: int = 4096,
    *,
    rule: str = "midpoint",
    modes: Sequence[str] = ("ideal", "kingston_noisy", "kingston_m3"),
    n_bootstrap: int = 1000,
    seed: int = 42,
    timestamp: Optional[str] = None,
) -> Dict[str, object]:
    ks = [int(k) for k in ks]
    at = angle_table(integrand, rule=rule)
    exact_a = exact_amplitude(at)
    exact_pks = {k: exact_pk(at, k) for k in ks}

    out: Dict[str, object] = {
        "integrand": {
            "name": integrand.name, "label": integrand.label, "kind": integrand.kind,
            "meta": {k: v for k, v in integrand.meta.items()},
        },
        "config": {
            "ks": ks, "shots": shots, "rule": rule, "n_index_qubits": 2, "n_qubits": 3,
            "opt_level": 0, "seed": seed, "modes": list(modes),
        },
        "exact_amplitude": exact_a,
        "exact_pk": exact_pks,
        "circuits": [], "modes": {}, "analysis": {"notes": []},
        "provenance": {"versions": _versions(),
                       "timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat(),
                       "note": "n=2 (4-point) QAE — pipeline validation, not a price or advantage."},
    }

    need_hw = any(m != "ideal" for m in modes)
    fake = B.kingston_backend() if need_hw else None
    ideal = B.ideal_sim() if "ideal" in modes else None
    if fake is not None:
        out["config"]["backend"] = fake.name
        out["config"]["backend_num_qubits"] = int(fake.num_qubits)

    # Pre-transpile the noisy circuits once; reuse counts + metadata across noisy/m3.
    transpiled: Dict[int, B.TranspiledCircuit] = {}
    noisy_counts: Dict[int, Dict[str, int]] = {}
    if fake is not None:
        for k in ks:
            tc = B.transpile_for(build_qae_circuit(at, k, measure=True), fake, optimization_level=0, seed=seed)
            transpiled[k] = tc
            noisy_counts[k] = B.run_counts(fake, tc.circuit, shots=shots, transpiled=True)
            out["circuits"].append({
                "k": k, "depth": tc.depth, "gate_counts": tc.gate_counts,
                "twoq_gates": tc.twoq_gates, "measurement_mapping": tc.measurement_mapping,
            })

    for mode in modes:
        succ: List[int] = []
        shot_list: List[int] = []
        mode_extra: Dict[str, object] = {}
        if mode == "ideal":
            for k in ks:
                counts = B.run_counts(ideal, build_qae_circuit(at, k, measure=True), shots=shots)
                s, n = B.ancilla_p1(counts)
                succ.append(s); shot_list.append(n)
        elif mode == "kingston_noisy":
            for k in ks:
                s, n = B.ancilla_p1(noisy_counts[k])
                succ.append(s); shot_list.append(n)
        elif mode == "kingston_m3":
            mit = MIT.build_mitigator(fake, transpiled[ks[0]].measurement_mapping)
            fids = []
            for k in ks:
                r = MIT.apply_m3(fake, noisy_counts[k], transpiled[k].measurement_mapping, shots, mitigator=mit)
                succ.append(r.successes_mitigated); shot_list.append(r.shots)
                fids = list(r.readout_fidelities)
            mode_extra["readout_fidelities"] = fids
            mode_extra["avg_readout_fidelity"] = (sum(fids) / len(fids)) if fids else None
        else:
            raise ValueError(f"unknown mode {mode!r}")

        res = M.estimate(ks, succ, shot_list, n_bootstrap=n_bootstrap, seed=seed)
        block = _mlae_block(res)
        block.update(mode_extra)
        block["abs_error_vs_exact"] = abs(res.a_hat - exact_a)
        out["modes"][mode] = block

    # Honest analysis: does dropping the deepest (noisiest) k help under noise?
    if "kingston_noisy" in out["modes"] and len(ks) >= 2:
        noisy = out["modes"]["kingston_noisy"]
        k01 = [i for i, k in enumerate(ks) if k in (0, 1)]
        if len(k01) >= 2:
            res01 = M.estimate([ks[i] for i in k01], [noisy["successes"][i] for i in k01],
                               [noisy["shots"][i] for i in k01], n_bootstrap=n_bootstrap, seed=seed)
            out["analysis"]["kingston_noisy_K01_only"] = {
                "a_hat": res01.a_hat, "abs_error_vs_exact": abs(res01.a_hat - exact_a),
                "ci_bootstrap": list(res01.ci_bootstrap),
            }

    return out


def _parse():
    p = argparse.ArgumentParser(description="Run a QAE campaign on the IBM Kingston simulator(s).")
    p.add_argument("--integrand", default="finance_call_spread",
                   help="g0|g1|g2|x|x2|finance_call_spread")
    p.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--rule", default="midpoint", choices=["left", "midpoint", "right"])
    p.add_argument("--modes", nargs="+", default=["ideal", "kingston_noisy", "kingston_m3"])
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--out", default="prototypes/qae/results")
    return p.parse_args()


def main():
    args = _parse()
    ig = get_integrand(args.integrand)
    res = run_campaign(ig, ks=args.ks, shots=args.shots, rule=args.rule,
                       modes=args.modes, n_bootstrap=args.n_bootstrap)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = res["provenance"]["timestamp_utc"].replace(":", "").replace("-", "")[:15]
    path = out_dir / f"qae_sim_{ig.name}_ks{''.join(map(str,args.ks))}_{stamp}.json"
    path.write_text(json.dumps(res, indent=2, default=str))

    ex = res["exact_amplitude"]
    print(f"integrand={ig.name}  exact_a={ex:.5f}  shots={args.shots}  ks={args.ks}")
    for mode, blk in res["modes"].items():
        lo, hi = blk["ci_bootstrap"]
        print(f"  {mode:16s} a_hat={blk['a_hat']:.5f}  err={blk['abs_error_vs_exact']:.5f}  "
              f"CI=[{lo:.4f},{hi:.4f}]")
    if "kingston_noisy_K01_only" in res["analysis"]:
        a = res["analysis"]["kingston_noisy_K01_only"]
        print(f"  (noisy K={{0,1}} only: a_hat={a['a_hat']:.5f} err={a['abs_error_vs_exact']:.5f})")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
