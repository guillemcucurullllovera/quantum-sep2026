"""Skip only source tests whose explicitly required private evidence is absent."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(items):
    scope = json.loads((ROOT / "PUBLIC_TEST_SCOPE.json").read_text())
    coverage = json.loads((ROOT / "SOURCE_COVERAGE.json").read_text())
    source_records = {row["path"]: row for row in coverage["source_inventory"]}
    for nodeid, required_paths in scope["excluded_tests"].items():
        for path in required_paths:
            record = source_records.get(path)
            if record is None or record["disposition"] != "EXCLUDED":
                raise pytest.UsageError(
                    f"Public test-scope input is not a documented source exclusion: {path}"
                )
        missing = [path for path in required_paths if not (ROOT / path).is_file()]
        if not missing:
            continue
        for item in items:
            if item.nodeid == nodeid:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            "requires excluded historical result/report inputs; "
                            "see PUBLIC_TEST_SCOPE.json"
                        )
                    )
                )
