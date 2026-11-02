"""IQAE-specific option-pricing entry point.

There is no separate IQAE option-pricing application distinct from the QAE
one in this codebase — ``applications.options.option_pricer`` is already
built on the IQAE estimator in ``algorithms.qae_iqae.iqae`` (not the
non-iterative MLAE variant), so "QAE" and "IQAE" name the same application
here. This module re-exports the same classes as ``applications.options.qae``
so both names resolve, and points at the underlying algorithm directly for
anyone who wants the estimator without the pricing wrapper.
"""
from algorithms.qae_iqae.iqae import run_iqae
from applications.options.option_pricer import DigitalOptionPricer, EuropeanOptionPricer

__all__ = ["DigitalOptionPricer", "EuropeanOptionPricer", "run_iqae"]
