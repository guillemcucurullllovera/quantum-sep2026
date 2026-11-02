"""Append-only local experiment registry, SQLite-backed (stdlib only — no
extra dependency for something this platform will query constantly).

Every ExperimentResult produced anywhere in the platform should be logged
here via `ExperimentRegistry.record(...)` so that benchmarks/ and
experiments/analysis/ have one consistent place to query history from,
instead of every algorithm module writing its own ad hoc results.json.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from core.config import get_settings
from core.errors import ExperimentNotFoundError
from core.schemas import ExperimentResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id TEXT PRIMARY KEY,
    algorithm TEXT NOT NULL,
    application TEXT,
    generation TEXT,
    adapted_from TEXT,
    created_at REAL NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_algorithm ON experiments(algorithm);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);
"""


class ExperimentRegistry:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_settings().registry_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(self, result: ExperimentResult) -> str:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiments "
                "(run_id, algorithm, application, generation, adapted_from, created_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result.metadata.run_id,
                    result.metadata.algorithm,
                    result.config.application,
                    result.metadata.generation,
                    result.metadata.adapted_from,
                    result.metadata.created_at,
                    result.model_dump_json(),
                ),
            )
        return result.metadata.run_id

    def get(self, run_id: str) -> ExperimentResult:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM experiments WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise ExperimentNotFoundError(f"No experiment with run_id={run_id!r}")
        return ExperimentResult.model_validate_json(row[0])

    def list_by_algorithm(self, algorithm: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM experiments WHERE algorithm = ? ORDER BY created_at DESC",
                (algorithm,),
            ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
