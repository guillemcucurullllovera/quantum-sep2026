"""Batch submission: run a list of circuits and return a list of
per-circuit outcomes, where one circuit's failure doesn't lose the results
already obtained for the others — this is the documented, deliberate
partial-failure behavior (fail soft per-item, not fail the whole batch).
"""
from __future__ import annotations

from dataclasses import dataclass

from core.schemas import BackendDescriptor
from hardware.ibm_runtime.primitives import SamplerRunResult, run_sampler


@dataclass
class BatchItemResult:
    index: int
    ok: bool
    result: SamplerRunResult | None = None
    error: str | None = None


@dataclass
class BatchResult:
    items: list[BatchItemResult]

    @property
    def n_succeeded(self) -> int:
        return sum(1 for i in self.items if i.ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for i in self.items if not i.ok)

    def successful_results(self) -> list[SamplerRunResult]:
        return [i.result for i in self.items if i.ok and i.result is not None]


def run_batch(circuits: list, backend_descriptor: BackendDescriptor, backend, shots: int = 4096) -> BatchResult:
    """Submit each circuit independently; a failure on circuit i is
    recorded as a BatchItemResult(ok=False) and does not stop the rest of
    the batch from running. Documented decision: partial failure is
    reported, never silently dropped and never allowed to abort sibling
    items — callers must check `n_failed` themselves.
    """
    items: list[BatchItemResult] = []
    for i, circuit in enumerate(circuits):
        try:
            result = run_sampler(circuit, backend_descriptor, backend, shots=shots)
            items.append(BatchItemResult(index=i, ok=True, result=result))
        except Exception as exc:  # noqa: BLE001 - intentional: isolate per-item failures
            items.append(BatchItemResult(index=i, ok=False, error=repr(exc)))
    return BatchResult(items=items)


__all__ = ["run_batch", "BatchResult", "BatchItemResult"]
