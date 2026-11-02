"""Unified job wrapper: a simulator "job" completes synchronously, a real
Runtime job is polled — callers use the same interface either way so
algorithm code doesn't need an if/else on backend kind.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    _result: object = field(default=None, repr=False)
    _status: JobStatus = JobStatus.QUEUED
    _error: Exception | None = field(default=None, repr=False)
    _backend_job: object = field(default=None, repr=False)  # real qiskit_ibm_runtime job, if any

    @classmethod
    def from_completed(cls, job_id: str, result: object) -> "Job":
        """Wrap an already-completed (simulator) result as a Job."""
        j = cls(job_id=job_id)
        j._result = result
        j._status = JobStatus.DONE
        return j

    @classmethod
    def from_runtime_job(cls, backend_job) -> "Job":
        """Wrap a real qiskit_ibm_runtime job — status/result are polled
        against the live job object."""
        return cls(job_id=backend_job.job_id(), _backend_job=backend_job)

    def status(self) -> JobStatus:
        if self._backend_job is None:
            return self._status
        raw = self._backend_job.status()
        mapping = {
            "QUEUED": JobStatus.QUEUED,
            "INITIALIZING": JobStatus.QUEUED,
            "VALIDATING": JobStatus.QUEUED,
            "RUNNING": JobStatus.RUNNING,
            "DONE": JobStatus.DONE,
            "ERROR": JobStatus.ERROR,
            "CANCELLED": JobStatus.CANCELLED,
        }
        return mapping.get(str(raw), JobStatus.RUNNING)

    def result(self, poll_interval_s: float = 1.0, timeout_s: float | None = None):
        if self._backend_job is None:
            if self._status == JobStatus.ERROR:
                raise self._error  # type: ignore[misc]
            return self._result
        start = time.monotonic()
        while self.status() not in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            if timeout_s is not None and (time.monotonic() - start) > timeout_s:
                raise TimeoutError(f"Job {self.job_id} did not finish within {timeout_s}s")
            time.sleep(poll_interval_s)
        if self.status() == JobStatus.ERROR:
            raise RuntimeError(f"Job {self.job_id} finished with ERROR status")
        return self._backend_job.result()


__all__ = ["Job", "JobStatus"]
