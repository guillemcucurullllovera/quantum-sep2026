"""Erosion audit — quantify, with real numbers, WHERE amplitude estimation does NOT help our book.

Three executed measurements (the honest negative that almost nobody publishes):

  A. QMC already reaches ~O(1/N) on our smooth payoffs.  Plain Monte-Carlo converges O(N^-1/2);
     amplitude estimation converges O(N^-1) — a quadratic edge OVER PLAIN MC. But randomized
     quasi-Monte-Carlo (scrambled Sobol) already reaches ~O(N^-1) classically on a low-dimensional
     smooth integrand — so AE's edge over the *best* classical method is ~zero. We measure the
     empirical convergence slope zeta for plain MC vs Sobol-QMC on the bull-call-spread payoff.

  B. Importance sampling erodes AE hardest exactly in the tail.  AE's quadratic is over PLAIN MC;
     for a rare crash-barrier event, classical importance sampling cuts the variance by a large
     factor, so the AE edge over the *variance-reduced* baseline shrinks. We measure the IS
     variance-reduction factor for a deep barrier probability.

  C. Herbert's loading cost cancels the AE edge for our log-concave distributions.  Preparing the
     lognormal via Grover-Rudolph needs classical partial-CDF integrals to O(1/eps) precision — the
     SAME 1/eps the AE query count saves. We tabulate the two O(1/eps) cost curves side by side.

This reproduces known results (Herbert 2021; the QMC/variance-reduction literature) APPLIED TO OUR
book, and is a benchmarking result, never an advantage claim.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy import integrate
from scipy.stats import norm, qmc

from algorithms.qae_iqae.finance_integrands import bs_lognormal_pdf, bs_call_price, bs_cdf_below


# --------------------------------------------------------------------------------------
# Shared BSM sampler
# --------------------------------------------------------------------------------------

def _S_of_Z(Z, S0, r, sigma, T):
    return S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * Z)


def _spread_payoff(S, K1, K2):
    return np.minimum(np.maximum(S - K1, 0.0), K2 - K1)


# --------------------------------------------------------------------------------------
# A. Plain MC vs Sobol-QMC convergence slope on the smooth spread payoff
# --------------------------------------------------------------------------------------

def audit_qmc_vs_mc(S0=100.0, K1=95.0, K2=115.0, r=0.04, sigma=0.20, T=0.25,
                    ms=(6, 8, 10, 12, 14), reps=40, seed=0) -> Dict:
    rng = np.random.default_rng(seed)
    # ground truth: high-accuracy quadrature of E[payoff] under the lognormal density
    true_val, _ = integrate.quad(
        lambda S: bs_lognormal_pdf(S, S0, r, sigma, T) * float(_spread_payoff(np.array([S]), K1, K2)[0]),
        1e-6, S0 * 6, limit=400)

    Ns = [2 ** m for m in ms]
    mc_rmse, qmc_rmse = [], []
    for N in Ns:
        mc_err, qmc_err = [], []
        for _ in range(reps):
            # plain MC
            Z = rng.standard_normal(N)
            est_mc = _spread_payoff(_S_of_Z(Z, S0, r, sigma, T), K1, K2).mean()
            mc_err.append(est_mc - true_val)
            # scrambled Sobol QMC (randomized -> RMSE meaningful)
            sob = qmc.Sobol(d=1, scramble=True, seed=rng)
            u = sob.random(N).ravel()
            u = np.clip(u, 1e-12, 1 - 1e-12)
            est_q = _spread_payoff(_S_of_Z(norm.ppf(u), S0, r, sigma, T), K1, K2).mean()
            qmc_err.append(est_q - true_val)
        mc_rmse.append(float(np.sqrt(np.mean(np.square(mc_err)))))
        qmc_rmse.append(float(np.sqrt(np.mean(np.square(qmc_err)))))

    def slope(rmse):
        return float(np.polyfit(np.log(Ns), np.log(rmse), 1)[0])

    return {
        "true_value": true_val, "Ns": Ns,
        "mc_rmse": mc_rmse, "qmc_rmse": qmc_rmse,
        "zeta_mc": slope(mc_rmse), "zeta_qmc": slope(qmc_rmse),
        "note": "zeta_mc ~ -0.5 (plain MC); zeta_qmc ~ -1 (Sobol) => AE's O(1/N) edge over the BEST "
                "classical method is ~zero on this smooth low-dim payoff.",
    }


# --------------------------------------------------------------------------------------
# B. Importance sampling vs plain MC on a deep (rare) crash-barrier probability
# --------------------------------------------------------------------------------------

def audit_importance_sampling(S0=100.0, r=0.04, sigma=0.20, T=0.25, crash_pct=0.25,
                              N=4096, reps=200, seed=1) -> Dict:
    rng = np.random.default_rng(seed)
    barrier = S0 * (1.0 - crash_pct)
    p_true = bs_cdf_below(barrier, S0, r, sigma, T)
    z_barrier = (math.log(barrier / S0) - (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))  # S_T<=B <=> Z<=z_b

    mc_est, is_est = [], []
    for _ in range(reps):
        # plain MC: fraction of samples in the tail
        Z = rng.standard_normal(N)
        mc_est.append(float(np.mean(Z <= z_barrier)))
        # importance sampling: shift sampling mean to the barrier, reweight by likelihood ratio
        mu = z_barrier
        Zs = rng.standard_normal(N) + mu
        w = np.exp(-mu * Zs + 0.5 * mu ** 2)  # N(0,1)/N(mu,1) likelihood ratio
        is_est.append(float(np.mean((Zs <= z_barrier) * w)))

    mc_rmse = float(np.sqrt(np.mean((np.array(mc_est) - p_true) ** 2)))
    is_rmse = float(np.sqrt(np.mean((np.array(is_est) - p_true) ** 2)))
    return {
        "crash_pct": crash_pct, "barrier": barrier, "p_true": p_true, "N": N, "reps": reps,
        "mc_rmse": mc_rmse, "is_rmse": is_rmse,
        "mc_rel_error": mc_rmse / p_true, "is_rel_error": is_rmse / p_true,
        "variance_reduction_factor": (mc_rmse / is_rmse) ** 2 if is_rmse > 0 else float("inf"),
        "note": "For a rare barrier, importance sampling cuts the classical variance by the factor "
                "shown; AE's quadratic edge is over PLAIN MC, so vs the variance-reduced baseline the "
                "edge is eroded exactly where tail risk lives.",
    }


# --------------------------------------------------------------------------------------
# C. Herbert's Grover-Rudolph loading cost vs the AE query budget (both O(1/eps))
# --------------------------------------------------------------------------------------

def audit_herbert_loading(epsilons=(1e-2, 1e-3, 1e-4)) -> Dict:
    """AE needs ~O(1/eps) oracle queries (IQAE Clopper-Pearson: N_oracle < 14/eps * log(...) ).
    Grover-Rudolph loading of a log-concave density needs the rotation angles theta_i = f(partial
    CDF integrals); to keep the loaded state accurate enough for an eps-accurate estimate, those
    integrals must be evaluated to ~O(1/eps) precision (Herbert 2021, PRE 103 063302). The classical
    preprocessing therefore scales O(1/eps) — the SAME order as the quantum saving => no net speedup."""
    rows = []
    for eps in epsilons:
        ae_queries = 14.0 / eps * math.log((2 / 0.05) * math.log2(math.pi / (4 * eps)))  # IQAE-CP bound
        gr_classical_ops = 1.0 / eps  # order of partial-CDF-integral precision (Herbert)
        rows.append({
            "epsilon": eps,
            "ae_oracle_queries_order": ae_queries,
            "grover_rudolph_classical_cdf_ops_order": gr_classical_ops,
            "ratio_same_order": ae_queries / gr_classical_ops,
        })
    return {
        "rows": rows,
        "note": "Both columns scale as 1/eps. The Grover-Rudolph loader does classical O(1/eps) CDF "
                "work of the same order as the AE query budget it was supposed to save (Herbert 2021). "
                "For our lognormal (log-concave) book, the loading step alone cancels the quadratic edge. "
                "NB: our current n=2/n=3 demo folds/loads the density classically anyway — it walks into "
                "this by construction.",
    }


# --------------------------------------------------------------------------------------

def run_all(seed=0) -> Dict:
    return {
        "A_qmc_vs_mc": audit_qmc_vs_mc(seed=seed),
        "B_importance_sampling": audit_importance_sampling(seed=seed + 1),
        "C_herbert_loading": audit_herbert_loading(),
        "provenance": {"note": "Executed classical benchmark (no quantum hardware). Reproduces "
                       "Herbert 2021 + QMC/variance-reduction results applied to our book. No advantage claim."},
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="quantum/qae/results/erosion_audit.json")
    args = ap.parse_args()
    res = run_all()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    A = res["A_qmc_vs_mc"]
    print(f"[A] smooth spread payoff, true E[payoff]={A['true_value']:.4f}")
    print(f"    plain-MC  slope zeta = {A['zeta_mc']:+.3f}  (expect ~ -0.50)")
    print(f"    Sobol-QMC slope zeta = {A['zeta_qmc']:+.3f}  (expect ~ -1.00)  -> AE edge ~zero")
    B = res["B_importance_sampling"]
    print(f"[B] deep barrier P(S_T<= {B['barrier']:.1f}) = {B['p_true']:.5f} (a {100*B['crash_pct']:.0f}% crash)")
    print(f"    plain-MC rel.err = {B['mc_rel_error']:.4f} ; importance-sampled rel.err = {B['is_rel_error']:.4f}")
    print(f"    variance reduction factor = {B['variance_reduction_factor']:.1f}x  -> AE edge eroded in the tail")
    print("[C] Grover-Rudolph loading cost vs AE queries (both ~1/eps):")
    for row in res["C_herbert_loading"]["rows"]:
        print(f"    eps={row['epsilon']:.0e}: AE queries~{row['ae_oracle_queries_order']:.3e}, "
              f"GR classical CDF ops~{row['grover_rudolph_classical_cdf_ops_order']:.3e}")
    print(f"    wrote {args.out}")


if __name__ == "__main__":
    main()
