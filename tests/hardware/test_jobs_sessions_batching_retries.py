import pytest

from hardware.jobs import Job, JobStatus
from hardware.retries import RetryExhaustedError, RetryPolicy, call_with_retry


def test_completed_job_returns_result_immediately():
    job = Job.from_completed("job-1", result={"counts": {"00": 1000}})
    assert job.status() == JobStatus.DONE
    assert job.result() == {"counts": {"00": 1000}}


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    sleeps = []
    result = call_with_retry(flaky, policy=RetryPolicy(max_attempts=5, base_delay_s=0.01), sleep=sleeps.append)
    assert result == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.01, 0.02]  # exponential backoff factor=2 by default


def test_retry_exhausts_and_raises_with_original_error_attached():
    def always_fails():
        raise ValueError("permanent")

    with pytest.raises(RetryExhaustedError) as exc_info:
        call_with_retry(always_fails, policy=RetryPolicy(max_attempts=3, base_delay_s=0.001), sleep=lambda s: None)
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_exception, ValueError)


def test_retry_respects_max_delay_cap():
    policy = RetryPolicy(max_attempts=10, base_delay_s=1.0, backoff_factor=10.0, max_delay_s=5.0)
    assert policy.delay_for_attempt(1) == 1.0
    assert policy.delay_for_attempt(5) == 5.0  # capped, would otherwise be 10000
