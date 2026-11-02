"""QUANTUM-1B hardware-readiness infrastructure.

Everything under this package is net-new engineering built to extend (never
modify) the frozen/shared `core.evidence`/`core.schemas`/`core.verification`
modules -- see `docs/quantum/QUANTUM1B_HARDWARE_READINESS.md` for the full
inventory of what is reused verbatim, what is extended here, and what
remains explicitly unfinished before this platform could responsibly run a
real IBM hardware job.

No module in this package ever makes a real network call, reads real IBM
Quantum credentials, or submits a real hardware job. Every test exercises
these modules exclusively against mock/fake/FakeBackend objects.
"""
