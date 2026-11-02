"""
Theoretical Sharpe metrics for the VALIDATED QRC daily results — task-spec (v3).

Supersedes compute_sharpe.py (v2); prior outputs preserved as *.v1.bak / *.v2.bak.
Differences from v2, per the explicit task spec:
  * Kelly uses the task formula   kelly_f = edge / (period_vol / sqrt(1)) * (1/ppy)
    (v2 used edge / (ann_vol / sqrt(5)) -> a ~10x larger number).
  * 1h->1D SPX uses per-period vol = 0.8% (v2 used 1.0%). At 0.8% the 10bps
    break-even BA = 56.25% and the 5bps break-even = 53.13%, matching the task's
    "need BA>=56% @10bps, BA>=53.2% @5bps".
  * Target-Sharpe table covers 1.0 / 1.5 / 2.0 (task) across tc = 10 / 5 / 2 bps.

Formula (long-short, 5-day rebalancing, 10 bps roundtrip), per task:
    edge                    = 2*BA - 1
    periods_per_year        = 252/5 = 50.4
    gross_return_per_period = edge * assumed_period_vol
    tc_per_period           = 0.10%   (10 bps roundtrip)
    net_return_per_period   = gross_return_per_period - tc_per_period
    ann_return              = net_return_per_period * periods_per_year
    ann_vol                 = assumed_period_vol * sqrt(periods_per_year)
    sharpe                  = ann_return / ann_vol            (NET; 'sharpe' == net)
    kelly_f                 = edge / (assumed_period_vol / sqrt(1)) * (1/periods_per_year)
We ALSO report gross_sharpe = edge*sqrt(ppy) (vol-independent, pre-cost). The task's
"sharpe ~= 0.40 ... matches IBM deck" refers to this GROSS/pre-cost figure (SPX gross
0.525, range 0.40-0.53); the NET sharpe after the 10bps roundtrip is only ~0.05 for SPX.

Validated daily BA are READ from synthesis/qrc_all_results.json -> summary.headline_results.
Per-period vols are task-specified assumptions. The synthesis file is left UNTOUCHED
(out of the stated task scope). Only benchmark/qrc_sharpe_metrics.json is written.

Usage: python3 compute_sharpe_v3.py
"""
import json
import math

BENCH_DIR = "."
SYNTH_JSON = f"{BENCH_DIR}/qrc_all_results.json"
ONEH_JSON = f"{BENCH_DIR}/qrc_1h_results.json"
OUT_JSON = f"{BENCH_DIR}/qrc_sharpe_metrics.json"

# ── READ existing results ──────────────────────────────────────────────────────
with open(SYNTH_JSON) as f:
    synth = json.load(f)
with open(ONEH_JSON) as f:
    oneh = json.load(f)

# Validated daily BA straight from the results file (summary.headline_results).
HEADLINE = {h["asset"]: h for h in synth["summary"]["headline_results"]}
NAME_MAP = {"GBP/USD": "GBPUSD", "SPX": "SPX", "NDX": "NDX", "Gold": "XAUUSD"}

# Task-specified assumptions. vol_pp_pct = assumed per-period (5-day) return std, in %.
# descr_ann_vol_pct = the task's parenthetical "annualized ~X%" (descriptive only —
# NOT used in any formula; the computed ann_vol = vol_pp*sqrt(ppy) is used instead).
ASSUMP = {
    "SPX":    {"vol_pp_pct": 1.5, "descr_ann_vol_pct": 17, "signal": "momentum"},
    "NDX":    {"vol_pp_pct": 1.8, "descr_ann_vol_pct": 20, "signal": "momentum"},
    "GBPUSD": {"vol_pp_pct": 0.7, "descr_ann_vol_pct": 7,  "signal": "momentum"},
    "XAUUSD": {"vol_pp_pct": 1.3, "descr_ann_vol_pct": 14, "signal": "mean-reversion (inverted)"},
}
ORDER = ["SPX", "NDX", "GBPUSD", "XAUUSD"]

TC_DAILY_BPS = 10
PPY_5D = 252 / 5  # = 50.4


# ── core formulas (task spec, literal) ─────────────────────────────────────────
def metrics(ba, vol_pp_pct, tc_bps, ppy):
    """Long-short theoretical metrics for one (BA, per-period vol, TC, ppy) point."""
    vol = vol_pp_pct / 100.0            # per-period return std (fraction)
    tc = tc_bps / 10000.0              # roundtrip cost per period (fraction)
    edge = 2.0 * ba - 1.0

    gross_pp = edge * vol
    net_pp = gross_pp - tc
    ann_ret = net_pp * ppy
    ann_gross = gross_pp * ppy
    ann_vol = vol * math.sqrt(ppy)

    net_sharpe = ann_ret / ann_vol
    gross_sharpe = ann_gross / ann_vol          # identity: == edge*sqrt(ppy)
    kelly_f = edge / (vol / math.sqrt(1)) * (1.0 / ppy)   # task formula, literal

    return {
        "ba": round(ba, 4),
        "edge_pct": round(edge * 100, 3),
        "vol_per_period_pct": round(vol_pp_pct, 3),
        "gross_ret_per_period_pct": round(gross_pp * 100, 4),
        "tc_per_period_pct": round(tc * 100, 4),
        "net_ret_per_period_pct": round(net_pp * 100, 4),
        "ann_return_pct": round(ann_ret * 100, 3),
        "ann_gross_return_pct": round(ann_gross * 100, 3),
        "ann_vol_pct": round(ann_vol * 100, 3),
        "gross_sharpe": round(gross_sharpe, 4),
        "net_sharpe": round(net_sharpe, 4),
        "sharpe": round(net_sharpe, 4),      # 'sharpe' == net, per the formula
        "kelly_f": round(kelly_f, 4),
    }


def breakeven_ba(vol_pp_pct, tc_bps):
    """BA at which net return per period = 0 (net Sharpe = 0)."""
    return round(0.5 + (tc_bps / 10000.0) / (2 * vol_pp_pct / 100.0), 4)


def ba_for_sharpe(target, vol_pp_pct, tc_bps, ppy):
    """BA needed to reach a target NET Sharpe.
    net_sharpe = (edge*vol - tc)*sqrt(ppy)/vol  ->  edge = tc/vol + S/sqrt(ppy)."""
    vol = vol_pp_pct / 100.0
    tc = tc_bps / 10000.0
    edge = tc / vol + target / math.sqrt(ppy)
    return round(0.5 + edge / 2.0, 4)


line = "=" * 82

# ═══════════════════════════════════════════════════════════════════════════════
# 1) DAILY VALIDATED ASSETS  (5-day rebalance, 10 bps, ppy = 50.4)
# ═══════════════════════════════════════════════════════════════════════════════
print(line)
print("QRC THEORETICAL SHARPE — validated daily results (2022-2026, 1583 test days)")
print("Long-short, 5-day rebalance, 10 bps roundtrip TC, ppy = 252/5 = 50.4")
print(line)
print(f"{'Asset':<8}{'BA':>7}{'p':>7} {'sig':<4}{'edge%':>7}{'gross/pp%':>10}"
      f"{'net/pp%':>9}{'annRet%':>9}{'grossSh':>9}{'netSh':>8}{'kelly_f':>9}")
print("-" * 82)

daily_out = {}
for key in ORDER:
    src_name = [k for k, v in NAME_MAP.items() if v == key][0]  # e.g. GBPUSD -> GBP/USD
    h = HEADLINE[src_name]
    a = ASSUMP[key]
    m = metrics(h["ba"], a["vol_pp_pct"], TC_DAILY_BPS, PPY_5D)
    m.update({
        "asset": src_name,
        "descriptive_annual_vol_pct": a["descr_ann_vol_pct"],
        "p": h["p"],
        "signal": a["signal"],
        "method": h["method"],
        "breakeven_ba": breakeven_ba(a["vol_pp_pct"], TC_DAILY_BPS),
    })
    daily_out[key] = m
    print(f"{src_name:<8}{h['ba']:>7.3f}{h['p']:>7.3f} {'':<4}"
          f"{m['edge_pct']:>6.1f}%{m['gross_ret_per_period_pct']:>9.4f}%"
          f"{m['net_ret_per_period_pct']:>8.4f}%{m['ann_return_pct']:>8.2f}%"
          f"{m['gross_sharpe']:>9.3f}{m['net_sharpe']:>8.3f}{m['kelly_f']:>9.4f}")

print("\ngross_sharpe = edge*sqrt(50.4)  (signal quality, pre-cost; vol-independent).")
print("  -> this is the task's 'sharpe ~= 0.40 (IBM deck)' figure: range 0.40-0.53,")
print(f"     SPX gross = {daily_out['SPX']['gross_sharpe']:.3f}.")
print("net_sharpe   = after the fixed 10 bps/period TC — dominates at ~53% BA,")
print(f"     SPX net = {daily_out['SPX']['net_sharpe']:.3f}  (ann_ret "
      f"{daily_out['SPX']['ann_return_pct']:.2f}%).")
print("break-even BA (net=0):  " + "  ".join(
    f"{daily_out[k]['asset']}={daily_out[k]['breakeven_ba']:.4f}" for k in ORDER))

# ═══════════════════════════════════════════════════════════════════════════════
# 2) 1h SPX at a 1-DAY horizon (rebalance daily, ppy=252, per-period vol=0.8%)
# ═══════════════════════════════════════════════════════════════════════════════
SPX_1H_VOL_PCT = 0.8       # task spec
PPY_1D = 252
TC_GRID = [10, 5, 2]       # 10bps retail/ETF ... 2bps liquid ES futures
BA_GRID_1H = [0.53, 0.54, 0.55, 0.56]

measured_1h_ba = oneh["results"]["SPX_1h"]["QRC_ridge_only"]["ba"]  # 0.4744 (mean-reversion)

print("\n" + line)
print("1h-DATA SPX at 1-DAY horizon — rebalance daily, ppy=252, per-period vol=0.8%")
print(f"(measured n=10 1h QRC BA = {measured_1h_ba:.4f} -> inverted {1-measured_1h_ba:.4f};")
print(" the 53-56% grid below is the projection/target BA, not the measured n=10 value)")
print(line)
print(f"{'BA':>6}{'edge%':>7}{'gross/day%':>12}{'grossSh':>9}   net Sharpe / ann_ret% @ TC")
print(f"{'':>34}   {'10bps':>15}{'5bps':>15}{'2bps':>15}")
print("-" * 82)

sharpe_1h = {}
for ba in BA_GRID_1H:
    row = {
        "ba": ba,
        "edge_pct": round((2 * ba - 1) * 100, 3),
        "gross_ret_per_period_pct": round((2 * ba - 1) * SPX_1H_VOL_PCT, 4),  # %/day
        "gross_sharpe": round((2 * ba - 1) * math.sqrt(PPY_1D), 4),
        "net_sharpe_by_tc_bps": {},
        "ann_return_pct_by_tc_bps": {},
        "net_ret_per_period_pct_by_tc_bps": {},
    }
    cells = []
    for tc in TC_GRID:
        m = metrics(ba, SPX_1H_VOL_PCT, tc, PPY_1D)
        row["net_sharpe_by_tc_bps"][str(tc)] = m["net_sharpe"]
        row["ann_return_pct_by_tc_bps"][str(tc)] = m["ann_return_pct"]
        row["net_ret_per_period_pct_by_tc_bps"][str(tc)] = m["net_ret_per_period_pct"]
        cells.append(f"{m['net_sharpe']:>7.3f}/{m['ann_return_pct']:>6.2f}%")
    sharpe_1h[f"BA_{ba:.2f}"] = row
    print(f"{ba:>6.2f}{row['edge_pct']:>6.1f}%{row['gross_ret_per_period_pct']:>11.3f}%"
          f"{row['gross_sharpe']:>9.3f}   " + "".join(f"{c:>15}" for c in cells))

be10 = breakeven_ba(SPX_1H_VOL_PCT, 10)
be5 = breakeven_ba(SPX_1H_VOL_PCT, 5)
print(f"\nBA=0.53 @10bps: gross {sharpe_1h['BA_0.53']['gross_ret_per_period_pct']:.3f}%/day, "
      f"net {sharpe_1h['BA_0.53']['net_ret_per_period_pct_by_tc_bps']['10']:.3f}%/day "
      f"(NEGATIVE after 10bps).")
print(f"Break-even BA @10bps = {be10:.4f}  (need BA>=56%).   "
      f"Break-even BA @5bps = {be5:.4f}  (need BA>=53.2%).")

# ═══════════════════════════════════════════════════════════════════════════════
# 3) BA REQUIRED for target NET Sharpe  ("what BA do we need for Sharpe 1.0/1.5/2.0")
# ═══════════════════════════════════════════════════════════════════════════════
TARGETS = [0.5, 1.0, 1.5, 2.0]
BE_TCS = [10, 5, 2]

print("\n" + line)
print("BA REQUIRED for target NET Sharpe, at each frequency (per-asset vol)")
print(line)

ba_required = {"daily_5day": {}, "1h_to_1d": {}}

# --- daily 5-day, per validated asset (each at its own assumed per-period vol) ---
print("\nDAILY 5-day rebalance (ppy=50.4):")
print(f"{'Asset (vol)':<18}{'TC':>6}{'breakeven':>11}{'Sh=1.0':>9}{'Sh=1.5':>9}{'Sh=2.0':>9}{'Sh=0.5':>9}")
print("-" * 82)
for key in ORDER:
    a = ASSUMP[key]
    label = f"{daily_out[key]['asset']}_vol{a['vol_pp_pct']}%"
    ba_required["daily_5day"][label] = {}
    for tc in BE_TCS:
        rec = {"breakeven": breakeven_ba(a["vol_pp_pct"], tc)}
        for t in TARGETS:
            rec[f"sharpe_{t}"] = ba_for_sharpe(t, a["vol_pp_pct"], tc, PPY_5D)
        ba_required["daily_5day"][label][f"tc_{tc}bps"] = rec
        print(f"{label:<18}{tc:>4}bp{rec['breakeven']:>11.4f}{rec['sharpe_1.0']:>9.4f}"
              f"{rec['sharpe_1.5']:>9.4f}{rec['sharpe_2.0']:>9.4f}{rec['sharpe_0.5']:>9.4f}")

# --- 1h -> 1D SPX (vol 0.8%, ppy 252) ---
print("\n1h -> 1-DAY horizon SPX (ppy=252, per-period vol=0.8%):")
print(f"{'vol':<18}{'TC':>6}{'breakeven':>11}{'Sh=1.0':>9}{'Sh=1.5':>9}{'Sh=2.0':>9}{'Sh=0.5':>9}")
print("-" * 82)
label_1h = f"SPX_vol{SPX_1H_VOL_PCT}%"
ba_required["1h_to_1d"][label_1h] = {}
for tc in BE_TCS:
    rec = {"breakeven": breakeven_ba(SPX_1H_VOL_PCT, tc)}
    for t in TARGETS:
        rec[f"sharpe_{t}"] = ba_for_sharpe(t, SPX_1H_VOL_PCT, tc, PPY_1D)
    ba_required["1h_to_1d"][label_1h][f"tc_{tc}bps"] = rec
    print(f"{label_1h:<18}{tc:>4}bp{rec['breakeven']:>11.4f}{rec['sharpe_1.0']:>9.4f}"
          f"{rec['sharpe_1.5']:>9.4f}{rec['sharpe_2.0']:>9.4f}{rec['sharpe_0.5']:>9.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE + SAVE  (only qrc_sharpe_metrics.json; synthesis file untouched)
# ═══════════════════════════════════════════════════════════════════════════════
output = {
    "meta": {
        "description": "Theoretical Sharpe & financial metrics for validated QRC daily signals (task-spec v3)",
        "formula": {
            "edge": "2*BA - 1",
            "periods_per_year_daily": "252/5 = 50.4",
            "gross_return_per_period": "edge * assumed_period_vol",
            "tc_per_period": "0.10% (10 bps roundtrip)",
            "net_return_per_period": "gross_return_per_period - tc_per_period",
            "ann_return": "net_return_per_period * periods_per_year",
            "ann_vol": "assumed_period_vol * sqrt(periods_per_year)",
            "sharpe": "ann_return / ann_vol  (NET; 'sharpe' field == net_sharpe)",
            "gross_sharpe": "edge * sqrt(periods_per_year)  (pre-cost, vol-independent)",
            "kelly_f": "edge / (assumed_period_vol / sqrt(1)) * (1/periods_per_year)",
            "ba_for_target_sharpe": "BA = 0.5 + (tc/vol + S/sqrt(ppy))/2",
        },
        "daily": {"rebalance": "5-day", "ppy": PPY_5D, "tc_bps_roundtrip": TC_DAILY_BPS},
        "ba_source": "synthesis/qrc_all_results.json -> summary.headline_results (validated daily BA)",
        "vol_source": "task-specified per-period (5-day) return std assumptions",
        "test_period": "2022-01-01 to 2026-07-01",
        "test_days": 1583,
        "note_gross_vs_net": (
            "gross_sharpe = edge*sqrt(ppy) is the pre-cost 'signal quality' number and is "
            "what the task's 'sharpe ~= 0.40, matches IBM deck' refers to (range 0.40-0.53; "
            "SPX 0.525). net_sharpe subtracts the fixed 10bps/period cost and is only ~0.05 "
            "for SPX at 53.7% BA — the fixed cost dominates the small per-period edge."
        ),
        "note_kelly": (
            "kelly_f follows the task's literal expression edge/(period_vol/sqrt(1))*(1/ppy); "
            "sqrt(1)=1 is a no-op as written. This differs from the prior file's kelly_fraction "
            "(edge/(ann_vol/sqrt(5)) ~10x larger); values are not comparable across the two."
        ),
        "supersedes": "compute_sharpe.py v2 output (preserved as qrc_sharpe_metrics.json.v2.bak)",
        "benchmark_date": "2026-07-02",
    },
    "daily_validated_sharpes": daily_out,
    "sharpe_1h_spx_1d_horizon": {
        "assumptions": {
            "per_period_vol_pct": SPX_1H_VOL_PCT,
            "ppy": PPY_1D,
            "ann_vol_pct": round(SPX_1H_VOL_PCT / 100 * math.sqrt(PPY_1D) * 100, 3),
            "tc_grid_bps": TC_GRID,
            "measured_1h_qrc_ba": round(measured_1h_ba, 4),
            "measured_1h_qrc_ba_inverted": round(1 - measured_1h_ba, 4),
            "note": ("BA grid is projection/target, not the measured n=10 1h value. Measured 1h "
                     "SPX QRC is mean-reverting (BA 0.4744, inverted 0.5256, p=0.011)."),
            "breakeven_ba_by_tc_bps": {str(tc): breakeven_ba(SPX_1H_VOL_PCT, tc) for tc in TC_GRID},
        },
        "by_ba": sharpe_1h,
    },
    "ba_required_for_target_sharpe": ba_required,
    "interpretation": {
        "gross_sharpe_range_daily": "0.40-0.53 — the pre-cost 'Sharpe ~0.4' (IBM deck) figure.",
        "net_sharpe_daily_10bps": (
            f"SPX {daily_out['SPX']['net_sharpe']:+.3f}, NDX {daily_out['NDX']['net_sharpe']:+.3f}, "
            f"Gold {daily_out['XAUUSD']['net_sharpe']:+.3f}, GBP/USD {daily_out['GBPUSD']['net_sharpe']:+.3f} "
            "— at ~53% BA on a 5-day book, the 10bps roundtrip erases the edge."
        ),
        "why_low": ("Per-period edge (edge*vol ~ 0.04-0.11%) is comparable to the fixed 10bps cost; "
                    "net return is the small difference, so net Sharpe collapses well below gross."),
        "one_h_1d_10bps": ("1h SPX at a 1-day horizon needs BA>=56.25% just to break even at 10bps "
                           "(BA=53% -> net -0.052%/day, net Sharpe ~ -1.0)."),
        "one_h_1d_5bps": "At 5bps the 1h->1D break-even drops to BA=53.13% (~53.2%).",
        "levers": ("(a) higher BA — n=100 QPU projection 55-65% daily; (b) realistic instrument TC "
                   "— liquid ES/NQ futures ~2bps not 10bps; (c) higher-vol underlyings amplify a "
                   "fixed edge above a fixed cost."),
        "caveat": ("Model charges TC every period; real turnover is lower because the directional "
                   "signal persists (does not flip every bar), so net Sharpe here is a conservative floor."),
        "synthesis_file_note": ("synthesis/qrc_all_results.json still carries the v2 financial_metrics "
                                "(kelly via ann_vol/sqrt(5); 1h vol 1.0%); it was left untouched as it is "
                                "outside this task's write scope. Re-sync only on request."),
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n" + line)
print(f"Saved -> {OUT_JSON}")
print("Synthesis file qrc_all_results.json left UNTOUCHED (out of task scope).")
print("DONE.")
