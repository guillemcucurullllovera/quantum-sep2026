"""QOKit-adapted QAOA infrastructure: fast diagonal-cost simulation and
Fourier-basis parameter transfer, adapted from JPMorgan Chase's QOKit
(arXiv:2309.04841, Apache-2.0). See THIRD_PARTY_NOTICES.md for exactly what was
adapted vs. built fresh, and LICENSES/NOTICE entry for QOKit.

This module does NOT rebuild portfolio/QUBO constraint construction —
that already exists in `algorithms.qaoa_qubo.qubo_builder`. It reuses it.
"""
