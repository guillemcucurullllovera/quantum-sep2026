"""Consolidate `run_sim.py`/`run_ibm.py` campaign artifacts into a markdown summary.

HISTORICAL PROTOTYPE, preserved as-is, self-contained.

  python -m prototypes.qae.summarize_results --results prototypes/qae/results --out prototypes/qae/RESULTS_SUMMARY.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def load(results_dir: str) -> List[Dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "qae_sim_*.json"))):
        with open(path) as f:
            d = json.load(f)
        d["_path"] = os.path.basename(path)
        out.append(d)
    # de-dup: keep the latest artifact per (integrand, ks)
    latest: Dict[str, Dict] = {}
    for d in out:
        key = f"{d['integrand']['name']}|{d['config']['ks']}"
        latest[key] = d  # sorted order → last wins
    return list(latest.values())


def render(records: List[Dict]) -> str:
    L: List[str] = []
    if not records:
        return "_No campaign artifacts found._\n"
    v = records[0]["provenance"]["versions"]
    be = records[0]["config"].get("backend", "n/a")
    beq = records[0]["config"].get("backend_num_qubits", "?")
    L.append("# QAE simulator results — raw summary\n")
    L.append(f"- **Backend (noise model):** `{be}` ({beq} qubits, Heron r3) via `AerSimulator.from_backend` "
             f"— this is the **IBM Kingston *simulator***, not the QPU (hardware run pending token).")
    L.append(f"- **Stack:** qiskit {v['qiskit']}, aer {v['qiskit_aer']}, ibm-runtime "
             f"{v['qiskit_ibm_runtime']}, mthree {v['mthree']}.")
    L.append("- **Register:** 3 qubits (2 index + 1 ancilla), n=2 grid (**4 points**). "
             "MLAE (QAE-without-QPE), `optimization_level=0`.")
    L.append("- **Honesty:** at n=2 this is a **pipeline / algorithm-validation demonstration, not a price "
             "and not an advantage**.\n")

    for d in records:
        ig = d["integrand"]
        L.append(f"\n## {ig['name']} — {ig['label']}")
        L.append(f"- exact amplitude `a = {d['exact_amplitude']:.5f}`, "
                 f"ks = {d['config']['ks']}, shots = {d['config']['shots']}")
        if ig["kind"] == "finance":
            m = ig["meta"]
            L.append(f"- finance meta: true BS spread price = **{float(m['true_bs_spread_price']):.4f}**, "
                     f"coarse 4-point grid price = {float(m['grid_price_estimate']):.4f} "
                     f"(⇒ ~{100*abs(float(m['grid_price_estimate'])-float(m['true_bs_spread_price']))/float(m['true_bs_spread_price']):.0f}% "
                     f"discretisation error — inherent to n=2, separate from any quantum error)")

        # circuit sizes
        if d.get("circuits"):
            L.append("\n| k | depth | 2q gates | total ops |")
            L.append("|---|-------|----------|-----------|")
            for c in d["circuits"]:
                tot = sum(c["gate_counts"].values())
                L.append(f"| {c['k']} | {c['depth']} | {c['twoq_gates']} | {tot} |")

        # modes
        L.append("\n| mode | a_hat | abs err | bootstrap 95% CI | Fisher SE |")
        L.append("|------|-------|---------|------------------|-----------|")
        for mode, blk in d["modes"].items():
            L.append(f"| {mode} | {blk['a_hat']:.5f} | {blk['abs_error_vs_exact']:.5f} | "
                     f"{_fmt_ci(blk['ci_bootstrap'])} | {blk['se_fisher']:.5f} |")
        a = d.get("analysis", {}).get("kingston_noisy_K01_only")
        if a:
            L.append(f"\n- _honest note:_ dropping the deepest circuit (noisy **K={{0,1}} only**): "
                     f"a_hat={a['a_hat']:.5f}, err={a['abs_error_vs_exact']:.5f} "
                     f"(vs full-K noisy err — deeper k can HURT under noise).")
        m3 = d["modes"].get("kingston_m3", {})
        if "avg_readout_fidelity" in m3 and m3["avg_readout_fidelity"] is not None:
            L.append(f"- M3 readout fidelity (mapped qubits): avg={m3['avg_readout_fidelity']:.4f}, "
                     f"per-qubit={[round(x,4) for x in m3.get('readout_fidelities', [])]}")
        # per-k p_hat
        L.append("\n| k | exact p_k | ideal p̂ | noisy p̂ | M3 p̂ |")
        L.append("|---|-----------|---------|---------|-------|")
        exact_pk = d.get("exact_pk", {})
        for i, k in enumerate(d["config"]["ks"]):
            row = [str(k), f"{exact_pk.get(str(k), exact_pk.get(k, float('nan'))):.4f}"]
            for mode in ("ideal", "kingston_noisy", "kingston_m3"):
                blk = d["modes"].get(mode)
                row.append(f"{blk['p_hat'][i]:.4f}" if blk else "—")
            L.append("| " + " | ".join(row) + " |")

    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="prototypes/qae/results")
    p.add_argument("--out", default="prototypes/qae/RESULTS_SUMMARY.md")
    args = p.parse_args()
    md = render(load(args.results))
    with open(args.out, "w") as f:
        f.write(md)
    print(f"wrote {args.out} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
