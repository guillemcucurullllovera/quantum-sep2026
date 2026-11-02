"""European/digital option pricing via amplitude estimation — Asymmetrica's own
implementation of the pattern qiskit-finance popularized (encode a payoff onto an
ancilla amplitude, then amplitude-estimate it), built on this platform's own
state-prep/MLAE primitives rather than qiskit-finance's circuit classes.

Fills a real gap: no option-pricing code existed anywhere in the Asymmetrica Quantum
estate before this module. See THIRD_PARTY_NOTICES.md.
"""
