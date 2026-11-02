"""End-to-end example: price a European call spread via QAE/MLAE, compared
against the closed-form Black-Scholes price. Run:
    python -m examples.qae_option_pricing_example
"""
from __future__ import annotations

from applications.options.option_pricer import EuropeanOptionPricer


def main() -> None:
    pricer = EuropeanOptionPricer()
    quote = pricer.price_spread(S0=100.0, K1=95.0, K2=115.0, r=0.04, sigma=0.20, T=0.25, shots=2048, seed=1)
    print("European call-spread pricing (QAE vs. Black-Scholes):")
    print(f"  quantum price:   {quote.quantum_price:.4f}")
    print(f"  classical price: {quote.classical_price:.4f}")
    print(f"  abs error:       {quote.abs_error:.4f}  ({quote.rel_error*100:.2f}% relative)")
    print(f"  note: {quote.note}")


if __name__ == "__main__":
    main()
