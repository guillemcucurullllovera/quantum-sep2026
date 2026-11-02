"""Filesystem artifact store, addressed by run_id. Keeps large numeric
artifacts (arrays, figures) out of the SQLite experiment registry, which
only stores metadata + pointers here.
"""
from __future__ import annotations

from pathlib import Path

from core.config import get_settings
from core.serialization import dumps, loads


class ResultStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().results_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_json(self, run_id: str, name: str, obj: object) -> str:
        path = self._run_dir(run_id) / f"{name}.json"
        path.write_text(dumps(obj, indent=2))
        return str(path)

    def load_json(self, run_id: str, name: str) -> object:
        path = self._run_dir(run_id) / f"{name}.json"
        return loads(path.read_text())

    def path_for(self, run_id: str, name: str) -> Path:
        """Return (creating parent dirs) a path for an arbitrary artifact
        (e.g. a matplotlib figure) that the caller writes itself."""
        return self._run_dir(run_id) / name

    def list_artifacts(self, run_id: str) -> list[str]:
        d = self._run_dir(run_id)
        return sorted(p.name for p in d.iterdir() if p.is_file())

    def artifact_exists(self, run_id: str, name: str) -> bool:
        """Resolve a `'{run_id}/{name}'`-style reference (as used by
        `core.schemas.ExperimentResult.financial_backtest_artifact`) against
        real, on-disk files — used by `core.evidence.compute_evidence_level`
        so a financial-edge claim (level G) requires an artifact that
        actually exists, not merely a non-empty string. Accepts either the
        bare name or the `.json` name `save_json` writes. Does not create
        `run_id`'s directory as a side effect of checking a (possibly
        forged) reference — unlike `list_artifacts`/`_run_dir`."""
        d = self.root / run_id
        if not d.is_dir():
            return False
        return (d / name).is_file() or (d / f"{name}.json").is_file()
