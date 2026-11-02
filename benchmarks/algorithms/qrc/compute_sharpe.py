"""
Theoretical Sharpe & financial metrics for the VALIDATED QRC daily results.

Inputs (validated daily balanced-accuracy, 2022-2026, 1583 test days):
    GBP/USD : BA = 53.0%  p=0.021  *    momentum
    SPX     : BA = 53.7%  p=0.005  **   momentum   (headline)
    NDX     : BA = 52.8%  p=0.033  *    momentum   (WITHDRAWN Q10/CSIM - not a quantum advantage)
    Gold    : BA = 52.8%  p=0.032  *    mean-reversion (signal inverted to trade)

Formula (long-short, 5-day rebalancing) — EXACTLY as specified for the IBM deck:
    edge                     = 2*BA - 1                       # skill beyond chance
    gross_return_per_period  = edge * assumed_vol_per_period
    tc_per_period            = 10 bps roundtrip = 0.10%
    net_return_per_period    = gross_return_per_period - tc_per_period
    periods_per_year         = 252 / 5 = 50.4
    ann_return               = net_return_per_period * periods_per_year
    ann_vol                  = assumed_vol_per_period * sqrt(periods_per_year)
    sharpe (net)             = ann_return / ann_vol
    kelly_fraction           = edge / (annualized_vol / sqrt(5))

We report BOTH the gross Sharpe (signal quality, pre-cost) and the net Sharpe
(after the 10 bps roundtrip). Note the identity  gross_sharpe = edge*sqrt(ppy),
independent of the vol assumption — the vol assumption only affects how badly the
fixed per-period transaction cost erodes the net return.

Usage: python3 compute_sharpe.py
Outputs:
    quantum_layer/benchmark/qrc_sharpe_metrics.json      (primary)
    quantum_layer/synthesis/qrc_all_results.json         ('financial_metrics' key appended, additive)
"""
import json
import math

BENCH_DIR = "."
SYNTH_JSON = f"{BENCH_DIR}/qrc_all_results.json"
OUT_JSON = f"{BENCH_DIR}/qrc_sharpe_metrics.json"

# ── Validated daily results ───────────────────────────────────────────────────
# vol_5d_pct  = assumed 5-day (per-period) return std, in %
# ann_vol_pct = annualized vol used only for the Kelly denominator (task-specified)
DAILY_RESULTS = {
    "GBPUSD": {"asset": "GBP/USD", "ba": 0.530, "vol_5d_pct": 0.70, "ann_vol_pct": 7.0,
               "p": 0.021, "sig": "★",       "signal": "momentum",
               "method": "QRC n=6 Ridge, 3 momentum features"},
    "SPX":    {"asset": "SPX",     "ba": 0.537, "vol_5d_pct": 1.50, "ann_vol_pct": 16.0,
               "p": 0.005, "sig": "★★", "signal": "momentum",
               "method": "QRC n=10 Ridge+concat, 8 macro features"},
    "NDX":    {"asset": "NDX",     "ba": 0.528, "vol_5d_pct": 1.80, "ann_vol_pct": 19.0,
               "p": 0.033, "sig": "★",       "signal": "momentum",
               "method": "QRC n=10 LGBM, 8 macro features, TB labels"},
    "XAUUSD": {"asset": "Gold",    "ba": 0.528, "vol_5d_pct": 1.30, "ann_vol_pct": 14.0,
               "p": 0.032, "sig": "★",       "signal": "mean-reversion (inverted)",
               "method": "QRC n=10 Ridge inverted, 8 commodity features"},
}

TC_DAILY_BPS = 10                 # 10 bps roundtrip per 5-day rebalance (task spec)
PERIODS_PER_YEAR_5D = 252 / 5     # = 50.4


def compute_metrics(ba, vol_pp_pct, ann_vol_pct, tc_bps, ppy):
    """Long-short theoretical metrics for one (BA, vol, TC, frequency) point."""
    vol = vol_pp_pct / 100.0            # per-period return std (fraction)
    tc = tc_bps / 10000.0               # roundtrip cost per period (fraction)
    edge = 2.0 * ba - 1.0               # directional skill beyond chance

    gross_pp = edge * vol               # gross return per period
    net_pp = gross_pp - tc              # net return per period

    ann_vol = vol * math.sqrt(ppy)      # annualized vol
    ann_gross_ret = gross_pp * ppy      # annualized gross return
    ann_net_ret = net_pp * ppy          # annualized net return

    gross_sharpe = ann_gross_ret / ann_vol if ann_vol > 0 else 0.0   # == edge*sqrt(ppy)
    net_sharpe = ann_net_ret / ann_vol if ann_vol > 0 else 0.0

    # Kelly per task spec: edge / (annualized_vol / sqrt(5))  — raw full-Kelly.
    kelly_denom = (ann_vol_pct / 100.0) / math.sqrt(5)
    kelly = edge / kelly_denom if kelly_denom > 0 else 0.0

    return {
        "ba": round(ba, 4),
        "edge_pct": round(edge * 100, 2),
        "vol_per_period_pct": round(vol_pp_pct, 3),
        "annualized_vol_pct": round(ann_vol_pct, 2),
        "gross_ret_per_period_pct": round(gross_pp * 100, 4),
        "tc_per_period_pct": round(tc * 100, 4),
        "net_ret_per_period_pct": round(net_pp * 100, 4),
        "ann_return_pct": round(ann_net_ret * 100, 2),
        "ann_gross_return_pct": round(ann_gross_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "gross_sharpe": round(gross_sharpe, 3),
        "net_sharpe": round(net_sharpe, 3),
        "sharpe": round(net_sharpe, 3),          # 'sharpe' == net (headline)
        "kelly_fraction": round(kelly, 3),
    }


def breakeven_ba(vol_pp_pct, tc_bps):
    """BA at which net return per period = 0 (net Sharpe = 0)."""
    vol = vol_pp_pct / 100.0
    tc = tc_bps / 10000.0
    return round(0.5 + tc / (2 * vol), 4)


def ba_for_sharpe(target_sharpe, vol_pp_pct, tc_bps, ppy):
    """BA needed to hit a target NET Sharpe."""
    vol = vol_pp_pct / 100.0
    tc = tc_bps / 10000.0
    ann_vol = vol * math.sqrt(ppy)
    req_net_pp = target_sharpe * ann_vol / ppy      # net return per period needed
    edge = (req_net_pp + tc) / vol
    return round(0.5 + edge / 2, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# 1) DAILY VALIDATED ASSETS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("QRC THEORETICAL SHARPE  —  Validated daily results (2022-2026, 1583 test days)")
print("Long-short, 5-day rebalance, 10 bps roundtrip TC, ppy = 252/5 = 50.4")
print("=" * 78)
print(f"\n{'Asset':<8}{'BA':>7}{'p':>7} {'sig':<3}{'edge%':>7}{'gross_Sh':>9}"
      f"{'net_Sh':>8}{'annRet%':>9}{'kelly':>7}")
print("-" * 78)

daily_out = {}
for key, d in DAILY_RESULTS.items():
    m = compute_metrics(d["ba"], d["vol_5d_pct"], d["ann_vol_pct"], TC_DAILY_BPS,
                        PERIODS_PER_YEAR_5D)
    m.update({"asset": d["asset"], "p": d["p"], "sig": d["sig"],
              "signal": d["signal"], "method": d["method"],
              "breakeven_ba": breakeven_ba(d["vol_5d_pct"], TC_DAILY_BPS)})
    daily_out[key] = m
    print(f"{d['asset']:<8}{d['ba']:>7.3f}{d['p']:>7.3f} {d['sig']:<3}"
          f"{m['edge_pct']:>6.1f}%{m['gross_sharpe']:>9.3f}{m['net_sharpe']:>8.3f}"
          f"{m['ann_return_pct']:>8.2f}%{m['kelly_fraction']:>7.3f}")

print("\nGross Sharpe = edge*sqrt(50.4)  (signal quality, pre-cost; vol-independent).")
print("Net Sharpe   = after 10 bps/period TC  (the fixed cost dominates at these BAs).")
print("Break-even BA (net=0):  " + "  ".join(
    f"{DAILY_RESULTS[k]['asset']}={daily_out[k]['breakeven_ba']:.3f}" for k in DAILY_RESULTS))

# ═══════════════════════════════════════════════════════════════════════════════
# 2) 1h SPX at a 1-DAY horizon (rebalance daily, ppy = 252), BA 53-55%
# ═══════════════════════════════════════════════════════════════════════════════
SPX_1D_VOL_PCT = 1.0      # ~1-day SPX std (annualized ~15.9%, matches SPX ~16% assumption)
PPY_1D = 252
TC_GRID_1H = [10, 5, 2]   # 10bps retail/ETF ... 2bps liquid ES futures
BA_GRID_1H = [0.53, 0.54, 0.55]

print("\n" + "=" * 78)
print("1h-DATA SPX at 1-DAY horizon  —  rebalance daily, ppy=252, 1D vol=1.0% (~16% ann.)")
print("=" * 78)
print(f"\n{'BA':>6}{'edge%':>7}{'gross_Sh':>10}  |   net Sharpe @ TC roundtrip")
print(f"{'':>23}   {'10bps':>8}{'5bps':>8}{'2bps':>8}")
print("-" * 78)

sharpe_1h = {}
for ba in BA_GRID_1H:
    row = {"ba": ba, "edge_pct": round((2 * ba - 1) * 100, 1),
           "gross_sharpe": round((2 * ba - 1) * math.sqrt(PPY_1D), 3),
           "net_sharpe_by_tc_bps": {}, "ann_return_pct_by_tc_bps": {}}
    cells = []
    for tc in TC_GRID_1H:
        m = compute_metrics(ba, SPX_1D_VOL_PCT, 16.0, tc, PPY_1D)
        row["net_sharpe_by_tc_bps"][str(tc)] = m["net_sharpe"]
        row["ann_return_pct_by_tc_bps"][str(tc)] = m["ann_return_pct"]
        cells.append(f"{m['net_sharpe']:>8.3f}")
    sharpe_1h[f"BA_{ba:.2f}"] = row
    print(f"{ba:>6.2f}{row['edge_pct']:>6.1f}%{row['gross_sharpe']:>10.3f}  |   "
          + "".join(cells))

print("\nAt 10 bps, daily rebalancing is TC-dominated (BA=55% only breaks even).")
print("At 2 bps (liquid ES futures), 1D-horizon SPX reaches Sharpe ~0.64 / 0.95 / 1.27")
print("for BA = 53 / 54 / 55%  —  higher than the daily 5-day book at equal BA,")
print("because ppy=252 gives sqrt(252)=15.9 vs sqrt(50.4)=7.1 (more independent bets).")

# ═══════════════════════════════════════════════════════════════════════════════
# 3) BREAK-EVEN / TARGET-BA TABLE (deck: "what BA do we need?")
# ═══════════════════════════════════════════════════════════════════════════════
configs = [
    {"label": "Daily 5D  (ppy=50.4, vol=1.5%)", "vol": 1.50, "ppy": PERIODS_PER_YEAR_5D},
    {"label": "1h -> 1D  (ppy=252,  vol=1.0%)", "vol": 1.00, "ppy": PPY_1D},
]
targets = [0.5, 1.0, 1.5]
breakeven_out = {}
print("\n" + "=" * 78)
print("BA REQUIRED for target NET Sharpe (SPX-like vol)")
print("=" * 78)
for c in configs:
    breakeven_out[c["label"]] = {}
    for tc in [10, 2]:
        reqs = {f"sharpe_{t}": ba_for_sharpe(t, c["vol"], tc, c["ppy"]) for t in targets}
        reqs["breakeven"] = breakeven_ba(c["vol"], tc)
        breakeven_out[c["label"]][f"tc_{tc}bps"] = reqs
        print(f"  {c['label']:<34} TC={tc:>2}bps  "
              f"be={reqs['breakeven']:.3f}  "
              f"Sh0.5={reqs['sharpe_0.5']:.3f}  "
              f"Sh1.0={reqs['sharpe_1.0']:.3f}  "
              f"Sh1.5={reqs['sharpe_1.5']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE + SAVE
# ═══════════════════════════════════════════════════════════════════════════════
output = {
    "meta": {
        "description": "Theoretical Sharpe & financial metrics for validated QRC daily signals",
        "formula": "long-short; edge=2*BA-1; net_ret_pp=edge*vol-tc; "
                   "sharpe=net_ret_pp*ppy/(vol*sqrt(ppy)); kelly=edge/(ann_vol/sqrt(5))",
        "daily": {"rebalance": "5-day", "ppy": PERIODS_PER_YEAR_5D, "tc_bps_roundtrip": TC_DAILY_BPS},
        "test_period": "2022-01-01 to 2026-07-01", "test_days": 1583,
        "note_gross_vs_net": "gross_sharpe = edge*sqrt(ppy) (vol-independent, signal quality). "
                             "net_sharpe subtracts fixed 10bps/period cost, which dominates at ~53% BA.",
        "benchmark_date": "2026-07-01",
    },
    "daily_validated_sharpes": daily_out,
    "sharpe_1h_spx_1d_horizon": {
        "assumptions": {"per_period_vol_pct": SPX_1D_VOL_PCT, "ppy": PPY_1D,
                        "annualized_vol_pct": 15.87, "tc_grid_bps": TC_GRID_1H},
        "by_ba": sharpe_1h,
    },
    "ba_required_for_target_sharpe": breakeven_out,
    "interpretation": {
        "gross_sharpe_range_daily": "0.40-0.53 (this is the 'Sharpe ~0.4' pre-cost figure)",
        "net_sharpe_daily_10bps": "SPX +0.05, NDX +0.00, Gold -0.15, GBP/USD -0.59 "
                                  "-- at ~53% BA on a 5-day book, 10bps roundtrip erases the edge.",
        "why_low": "Per-period edge (edge*vol ~ 0.04-0.11%) is comparable to the 10bps cost; "
                   "net return is the small difference.",
        "levers": "(a) higher BA -- n=100 QPU projection 55-65% daily; "
                  "(b) realistic instrument TC -- liquid ES/NQ futures ~2bps not 10bps; "
                  "(c) higher-vol underlyings amplify the same edge above a fixed cost.",
        "one_h_1d_headline": "1h data at a 1-day horizon, 2bps futures TC: net Sharpe "
                             "0.64/0.95/1.27 at BA 53/54/55% -- daily rebalancing multiplies "
                             "a given edge by sqrt(252)=15.9 in gross Sharpe.",
        "caveat": "Simple model charges TC every period; real turnover is lower because the "
                  "directional signal persists (does not flip every bar), so net Sharpe is a "
                  "conservative floor.",
    },
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved primary metrics -> {OUT_JSON}")

# Additive append to the synthesis file (preserve all existing content).
with open(SYNTH_JSON) as f:
    allres = json.load(f)
allres["financial_metrics"] = output
with open(SYNTH_JSON, "w") as f:
    json.dump(allres, f, indent=2, ensure_ascii=False)
print(f"Appended 'financial_metrics' key -> {SYNTH_JSON}")
print("\nDONE.")
