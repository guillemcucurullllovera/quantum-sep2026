# QRC benchmark scripts (historical)

These are real experiment-harness scripts that produced the QRC
hourly/multi-timeframe realized-volatility benchmark results referenced
in `docs/research/`.

**Import path fix applied** (mechanical only, no logic changes): these
scripts originally imported `quantum.components.qrc.{qrc_v1,qrc_encoder}`.
That exact module was never physically preserved elsewhere in this repo (the
`algorithms/qrc/reservoir.py` is a *refactor*, not a verbatim port —
different function signatures), so it has been ported verbatim to
`algorithms/qrc/gen2_v1_original/` and the imports repointed there. This
keeps these historical benchmarks running against the exact code that
produced their original results, rather than against a later reimplementation.

**Not runnable offline, by design, not a bug**: `compute_sharpe.py`,
`compute_sharpe_v3.py`, `qrc_1h_bench_v5.py`, `qrc_1h_benchmark.py`,
`qrc_1h_qr2.py`, `qrc_1h_v2.py`, `qrc_1h_v2_fast.py`, `qrc_1h_v3.py`,
`qrc_multifreq_benchmark.py` fetch real market data via `yfinance`, which is
neither installed in this project's venv nor reachable without network
access in this build environment. Install it (`pip install yfinance`) and
run with network access to actually execute them.

**Runnable as-is (verified by import currently)**: `compute_sharpe.py`,
`compute_sharpe_v3.py`, `qrc_m15_5d_benchmark.py`, `qrc_m15_v2.py`.

**RETIRED, not runnable — see `_RETIRED_ADVANTAGE_CLAIMS/RETIRED.md`**:
`make_slides.py` and `qrc_investor_demo.ipynb` were moved to
`_RETIRED_ADVANTAGE_CLAIMS/` during QUANTUM-1A (2026-09-30): both asserted
unexecuted, unsupported quantum-advantage headline numbers repudiated by
this platform's own Q10 audit. `make_slides.py` now raises `RuntimeError`
immediately on execution rather than "importing cleanly" as an earlier
version of this README stated — that description is corrected here per an
independent adversarial review (see
`docs/quantum/QUANTUM1A_REVIEW_REMEDIATION.md`, finding P1-3).

**Deliberately not ported**: the `*_results.json` files these scripts
produced (`qrc_1h_results.json`, `qrc_1h_v2_results.json`,
`qrc_1h_v3_results.json`, `qrc_1h_v5_results.json`,
`qrc_m15_v2_results.smoke.json`, `qrc_multifreq_results.json`,
`qrc_sharpe_metrics.json`) — generated output, not source, and not tracked
here.
