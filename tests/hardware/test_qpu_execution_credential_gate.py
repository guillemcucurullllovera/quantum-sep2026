"""The single most important test in the hardware/ layer: confirms the
platform NEVER silently substitutes a simulator for a requested QPU run,
and NEVER attempts a network call before checking for a credential.

`budget_tracker`/`session_id` became required keyword-only parameters of
`execute_on_qpu` when remediating independent-review finding P1-1 (budget
enforcement was previously completely unwired from this function). Every
call below passes a trivial, never-actually-invoked fake tracker — the
credential check fires, and `execute_on_qpu` raises, BEFORE
`budget_tracker.record_submission(...)` is ever reached, so these tests'
original intent (prove the credential check happens before anything else)
is preserved exactly; the fake only exists to satisfy the new required
parameter, never to be exercised.
"""
import pytest

from core.errors import CredentialError
from hardware.qpu_execution import execute_on_qpu


class _NeverCalledBudgetTracker:
    """A budget tracker that fails the test loudly if it is ever actually
    invoked — these tests must never get far enough to reach it."""

    def record_submission(self, session_id, shots, *, qpu_seconds=0.0, usd_cost=0.0):
        raise AssertionError(
            "budget_tracker.record_submission was called, meaning execute_on_qpu did "
            "NOT raise CredentialError before touching the budget tracker -- the exact "
            "ordering these tests exist to guard."
        )


def test_qpu_execution_raises_credential_error_with_no_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    import core.config as config_mod

    config_mod.get_settings(refresh=True)

    with pytest.raises(CredentialError, match="IBM_QUANTUM_TOKEN"):
        # circuit=None: proves the failure happens before the circuit is
        # ever touched, i.e. before any real submission logic runs.
        execute_on_qpu(
            None, backend_name="ibm_torino",
            budget_tracker=_NeverCalledBudgetTracker(), session_id="test-session",
        )


def test_credential_error_message_tells_the_user_what_to_do(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    import core.config as config_mod

    config_mod.get_settings(refresh=True)

    with pytest.raises(CredentialError) as exc_info:
        execute_on_qpu(
            None, backend_name="ibm_torino",
            budget_tracker=_NeverCalledBudgetTracker(), session_id="test-session",
        )
    msg = str(exc_info.value)
    assert ".env.example" in msg
    assert "simulat" in msg.lower()  # points the user at the no-credential-needed path


def test_budget_tracker_and_session_id_are_required():
    """Regression guard for the P1-1 remediation itself: calling
    execute_on_qpu WITHOUT budget_tracker/session_id must fail immediately
    (TypeError, from Python's own required-keyword-argument binding, which
    happens before the function body runs at all — independent of
    credential state) -- there must be no code path that reaches a real
    submission attempt without them."""
    with pytest.raises(TypeError, match="budget_tracker|session_id"):
        execute_on_qpu(None, backend_name="ibm_torino")
