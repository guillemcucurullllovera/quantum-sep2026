"""Amazon Braket integration layer.

Wraps the vendored SDK at `algorithms_extended.amazon_braket_asymmetrica.braket`
with this platform's credential/config/experiment-tracking conventions, so
Braket becomes a second backend target alongside the IBM-centric `hardware/`
package. See `hardware/braket/config.py` for credential handling and
`hardware/braket/backend_adapter.py` for the circuit-translation boundary.
"""
