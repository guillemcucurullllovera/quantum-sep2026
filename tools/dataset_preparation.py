"""python -m tools.dataset_preparation --input prices.csv --output prepared.csv

CLI front door to core.feature_engineering: reads a CSV with a single price
column, applies fractional differentiation (stationarity-preserving) and
triple-barrier labeling, writes the result. Not a new algorithm -- a
runnable pipeline composing what already exists, so dataset prep is a
one-command step instead of a hand-written script every time.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from core.feature_engineering.fracdiff import fracdiff_df, find_min_d
from core.feature_engineering.triple_barrier import get_daily_vol, triple_barrier_labels


def prepare(
    prices: pd.Series,
    *,
    pt: float = 0.02,
    sl: float = 0.01,
    max_holding: int = 10,
) -> pd.DataFrame:
    d = find_min_d(prices)
    frac_col = f"price_frac{d}"
    stationary = fracdiff_df(prices.to_frame("price"), columns=["price"], d=d)[frac_col]
    vol = get_daily_vol(prices)
    t1 = prices.index.to_series().shift(-max_holding).reindex(prices.index)
    events = prices.iloc[:-max_holding]  # a Series indexed by event start time, as required
    labels = triple_barrier_labels(prices, events=events, pt=pt, sl=sl, t1=t1, dynamic_width=True, vol=vol)
    out = pd.DataFrame({"price": prices, "stationary": stationary})
    out["label"] = labels.reindex(out.index)
    out.attrs["fracdiff_d"] = d
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.dataset_preparation")
    parser.add_argument("--input", required=True, help="CSV with a 'price' column, indexed by date")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-holding", type=int, default=10)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input, index_col=0, parse_dates=True)
    if "price" not in df.columns:
        print(f"Input must have a 'price' column, got: {list(df.columns)}", file=sys.stderr)
        return 1

    prepared = prepare(df["price"], max_holding=args.max_holding)
    prepared.to_csv(args.output)
    print(f"Wrote {len(prepared)} rows to {args.output} (fracdiff d={prepared.attrs.get('fracdiff_d'):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
