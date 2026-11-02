"""IBM Quantum / Qiskit Runtime hardware infrastructure layer.

Nothing in this package reads a credential except `core.config.Settings.ibm`,
and only `hardware.qpu_execution` ever calls `.require()` on it — every
other module here works identically against `qiskit_aer` simulators and
Qiskit fake backends, with no network access and no credential needed.
"""
