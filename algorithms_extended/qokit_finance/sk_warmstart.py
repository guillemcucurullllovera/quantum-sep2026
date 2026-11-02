"""SK-model (Sherrington-Kirkpatrick) transferable initial QAOA angles.

Data asset `assets/best_SK_QAOA_parameters.json` copied verbatim from
QOKit (`qokit/assets/best_SK_QAOA_parameters.json`, Apache-2.0, JP Morgan
Chase & Co) -- 93KB, 34 precomputed (gamma, beta) schedules at depths
p=1..24, empirically optimized for the infinite-size Sherrington-Kirkpatrick
model. This is a genuinely reusable, problem-class-agnostic warm start:
SK-optimal angles are known empirically to transfer reasonably well to
other combinatorial QUBO instances (including portfolio-selection QUBOs)
as a starting point for local optimization, which is why QOKit ships it
as a general default rather than a problem-specific lookup.

This is data-asset provenance, not code adaptation -- the loading/lookup
logic below is a small, independent reimplementation (not a copy of
QOKit's `_get_sk_gamma_beta`), reusing `parameter_basis.py`'s Fourier
extrapolation (already adapted in this directory) for depths not present
in the table, rather than duplicating that logic here too.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from algorithms_extended.qokit_finance.parameter_basis import transfer_parameters

_ASSET_PATH = Path(__file__).parent / "assets" / "best_SK_QAOA_parameters.json"


def _load_table() -> dict:
    with open(_ASSET_PATH) as f:
        return json.load(f)


def sk_gamma_beta(p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (gamma, beta) initial angles for QAOA depth p.

    If p is present in the precomputed table, returns it directly. If not,
    extrapolates from the nearest available p via `transfer_parameters`.
    """
    table = _load_table()
    if str(p) in table:
        row = table[str(p)]
        return np.array(row["gammas"]), np.array(row["betas"])

    available = sorted(int(k) for k in table)
    closest = min(available, key=lambda k: abs(k - p))
    row = table[str(closest)]
    gamma, beta = np.array(row["gammas"]), np.array(row["betas"])
    return transfer_parameters(gamma, beta, target_p=p)


def available_depths() -> list[int]:
    return sorted(int(k) for k in _load_table())
