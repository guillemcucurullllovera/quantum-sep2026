"""Tests for `experiments.quantum1b.hardware_readiness.dry_run_runbook` --
the end-to-end assembled-procedure dry run, closing
`PRE_HARDWARE_QUANTUM_CAPSTONE.md` §4 item 8 ("the runbook has never been
dry-run, even against a fake service end-to-end")."""
from __future__ import annotations

import pytest

from core.evidence import compute_evidence_level
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    HardwareRunEvidence,
    RunMetadata,
)
from experiments.quantum1b.hardware_readiness.budgets import BudgetExceededError
from experiments.quantum1b.hardware_readiness.frozen_controls import FROZEN_CONTROLS
from experiments.quantum1b.hardware_readiness.dry_run_runbook import (
    DryRunStepResult,
    run_dry_run_runbook,
)
from experiments.quantum1b.hardware_readiness.evidence_store import PersistentEvidenceStore


class TestDryRunRunbookHappyPath:
    def test_runs_one_step_per_frozen_control(self, tmp_path):
        results = run_dry_run_runbook(
            evidence_db_path=tmp_path / "dry_run_evidence.db",
            budget_db_path=tmp_path / "dry_run_budget.db",
        )
        assert len(results) == len(FROZEN_CONTROLS)
        for r in results:
            assert isinstance(r, DryRunStepResult)
            assert r.evidence_level_reached == "F"
            assert r.backend_selected == "ibm_kingston"

    def test_budget_increments_monotonically_across_steps(self, tmp_path):
        results = run_dry_run_runbook(
            evidence_db_path=tmp_path / "dry_run_evidence.db",
            budget_db_path=tmp_path / "dry_run_budget.db",
        )
        job_counts = [r.budget_snapshot_jobs_submitted for r in results]
        assert job_counts == sorted(job_counts)
        assert job_counts[-1] == len(FROZEN_CONTROLS)

    def test_every_attestation_is_durably_recorded_and_resolves_to_level_f(self, tmp_path):
        evidence_db = tmp_path / "dry_run_evidence.db"
        results = run_dry_run_runbook(evidence_db_path=evidence_db, budget_db_path=tmp_path / "dry_run_budget.db")

        # reopen the store as a FRESH object (simulating a different later
        # process / report generator) and confirm every attestation from
        # the dry run is independently resolvable and actually grants
        # evidence level F through the real core.evidence machinery.
        reopened_store = PersistentEvidenceStore(evidence_db)
        for r in results:
            attestation = reopened_store.get(r.attestation_id)
            assert attestation is not None

            result = ExperimentResult(
                metadata=RunMetadata(algorithm="dry_run"),
                config=ExperimentConfig(
                    name="dry_run",
                    algorithm="dry_run",
                    backend=BackendDescriptor(name=r.backend_selected, kind=BackendKind.QPU, num_qubits=156),
                ),
                hardware_evidence=HardwareRunEvidence(
                    backend_name=r.backend_selected,
                    job_id=f"dry-run-job-{r.family}-{r.n_candidates}",
                    execution_timestamp="2026-09-30T12:00:00+00:00",
                    attestation_id=r.attestation_id,
                ),
            )
            assert compute_evidence_level(result, evidence_store=reopened_store).value == "F"


class TestDryRunRunbookRefusesCorrectly:
    def test_refuses_when_budget_too_small_for_all_controls(self, tmp_path, monkeypatch):
        """If the budget can't fit every frozen control, the dry run must
        raise (via BudgetExceededError bubbling up from
        PersistentBudgetTracker.record_submission), never silently submit
        a subset and report success."""
        import experiments.quantum1b.hardware_readiness.dry_run_runbook as mod

        # Patch the budget construction to allow only 1 job, fewer than
        # len(FROZEN_CONTROLS) (3) -- forces a refusal partway through.
        from experiments.quantum1b.hardware_readiness.budgets import HardwareDayBudget

        original_get_or_create = mod.PersistentBudgetTracker.get_or_create_session

        def patched(self, session_id, budget):
            tiny_budget = HardwareDayBudget(
                max_jobs=1, max_shots_per_job=4096, max_total_shots=4096, max_qpu_seconds=600.0
            )
            return original_get_or_create(self, session_id, tiny_budget)

        monkeypatch.setattr(mod.PersistentBudgetTracker, "get_or_create_session", patched)

        with pytest.raises(BudgetExceededError):
            run_dry_run_runbook(evidence_db_path=tmp_path / "dry_run_evidence.db", budget_db_path=tmp_path / "dry_run_budget.db")

    def test_refuses_without_credentials(self, tmp_path, monkeypatch):
        import experiments.quantum1b.hardware_readiness.dry_run_runbook as mod

        def always_no_credentials(candidates, criteria, *, credentials_present):
            from experiments.quantum1b.hardware_readiness.budgets import select_backend as real_select
            return real_select(candidates, criteria, credentials_present=False)

        monkeypatch.setattr(mod, "select_backend", always_no_credentials)

        with pytest.raises(RuntimeError, match="backend selection refused"):
            run_dry_run_runbook(evidence_db_path=tmp_path / "dry_run_evidence.db", budget_db_path=tmp_path / "dry_run_budget.db")


class TestDryRunPathMarkerGuard:
    """Regression tests for independent-review finding P2-2 ('VerifiedAttestation
    has no provenance marker; fake dry-run evidence is structurally
    indistinguishable from real Level-F evidence once persisted'). The
    in-scope fix: refuse to run a dry run against any db path that doesn't
    contain the literal substring "dry_run", preventing the accidental-
    collision scenario the finding describes."""

    def test_refuses_an_evidence_path_without_the_marker(self, tmp_path):
        with pytest.raises(ValueError, match="dry_run"):
            run_dry_run_runbook(
                evidence_db_path=tmp_path / "evidence.db",  # no marker
                budget_db_path=tmp_path / "dry_run_budget.db",
            )

    def test_refuses_a_budget_path_without_the_marker(self, tmp_path):
        with pytest.raises(ValueError, match="dry_run"):
            run_dry_run_runbook(
                evidence_db_path=tmp_path / "dry_run_evidence.db",
                budget_db_path=tmp_path / "budget.db",  # no marker
            )

    def test_a_path_that_could_plausibly_be_a_real_hardware_day_session_is_refused(self, tmp_path):
        """The actual collision scenario P2-2 describes: a dry run
        accidentally pointed at what looks like a real Hardware Day
        session's store."""
        with pytest.raises(ValueError):
            run_dry_run_runbook(
                evidence_db_path=tmp_path / "hardware_day_2026-10-01_evidence.db",
                budget_db_path=tmp_path / "hardware_day_2026-10-01_budget.db",
            )

    def test_accepts_paths_with_the_marker(self, tmp_path):
        results = run_dry_run_runbook(
            evidence_db_path=tmp_path / "dry_run_evidence.db",
            budget_db_path=tmp_path / "dry_run_budget.db",
        )
        assert len(results) == len(FROZEN_CONTROLS)


class TestBudgetCheckedBeforeJobRegistration:
    """Regression test for independent-review finding P2-10 ('the dry-run
    pipeline registers the fake job before checking budget, inverting its
    own documented order')."""

    def test_budget_exceeded_error_fires_before_any_fake_job_is_registered(self, tmp_path, monkeypatch):
        import experiments.quantum1b.hardware_readiness.dry_run_runbook as mod
        from experiments.quantum1b.hardware_readiness.budgets import HardwareDayBudget

        original_get_or_create = mod.PersistentBudgetTracker.get_or_create_session

        def patched(self, session_id, budget):
            tiny_budget = HardwareDayBudget(
                max_jobs=0 + 1, max_shots_per_job=4096, max_total_shots=4096, max_qpu_seconds=600.0
            )
            return original_get_or_create(self, session_id, tiny_budget)

        monkeypatch.setattr(mod.PersistentBudgetTracker, "get_or_create_session", patched)

        # Spy on FakeDryRunService.job registration to confirm it is never
        # reached for the cell(s) that exceed budget.
        registered_job_ids: list[str] = []
        original_init = mod.FakeDryRunJob.__init__

        def spying_init(self, *args, **kwargs):
            registered_job_ids.append(kwargs.get("_job_id") or args[0])
            return original_init(self, *args, **kwargs)

        monkeypatch.setattr(mod.FakeDryRunJob, "__init__", spying_init)

        with pytest.raises(BudgetExceededError):
            run_dry_run_runbook(
                evidence_db_path=tmp_path / "dry_run_evidence.db",
                budget_db_path=tmp_path / "dry_run_budget.db",
            )

        # Exactly 1 job fit the tiny budget -- registered_job_ids must
        # contain AT MOST that many entries, never one for a cell whose
        # budget check actually failed (which would mean the job was
        # registered BEFORE the budget check, the P2-10 bug).
        assert len(registered_job_ids) <= 1


class TestDryRunRunbookNeverTouchesRealNetwork:
    def test_module_does_not_import_qiskit_runtime_service(self):
        import ast

        import experiments.quantum1b.hardware_readiness.dry_run_runbook as mod

        tree = ast.parse(open(mod.__file__).read())
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
        assert not any("qiskit_ibm_runtime" in name for name in imported_names), imported_names
