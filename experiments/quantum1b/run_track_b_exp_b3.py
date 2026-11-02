"""Track B runner: EXP-B3 -- Q10-style CSIM reproduction for CVaR-QAOA.

See experiments/quantum1b/q10_qaoa_reproduction.py for the full
implementation and methodology notes.

Run with: `python3 -m experiments.quantum1b.run_track_b_exp_b3` from the
repo root.

**Host-resource discipline (added 2026-10-01, overnight recovery pass after
an OOM kill on this shared host -- see q10_qaoa_reproduction.py's module
docstring for the full incident record):** BLAS thread counts are pinned to
a small fixed value BEFORE numpy/scipy are imported anywhere in the process
(env vars have no effect once those libraries have already loaded their
threading backend), to bound per-process thread-local workspace overhead
while this host concurrently runs other heavy agent sessions. This must
stay the first thing this module does.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

from experiments.quantum1b.q10_qaoa_reproduction import main  # noqa: E402

if __name__ == "__main__":
    main()
