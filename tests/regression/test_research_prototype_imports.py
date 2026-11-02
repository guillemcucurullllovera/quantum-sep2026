"""Import smoke tests and a secret-scanning regression guard for the
research-prototype packages under prototypes/pqk/p0_1_qrbf_ocsvm_replication/,
prototypes/phase0_research/, and prototypes/phase1_research/.
"""
import json

import pytest


def test_p0_1_qrbf_ocsvm_kernels_module_imports():
    from prototypes.pqk.p0_1_qrbf_ocsvm_replication import _kernels

    assert hasattr(_kernels, "FEATURE_MAPS")
    assert isinstance(_kernels.FEATURE_MAPS, (list, tuple, dict))


def test_p0_1_qrbf_ocsvm_data_module_imports():
    from prototypes.pqk.p0_1_qrbf_ocsvm_replication import _data

    assert _data is not None


def test_p0_1_replication_report_exists_and_nonempty():
    from pathlib import Path

    report = (
        Path(__file__).resolve().parents[2]
        / "prototypes"
        / "pqk"
        / "p0_1_qrbf_ocsvm_replication"
        / "REPLICATION_REPORT.md"
    )
    assert report.is_file()
    assert report.stat().st_size > 100


def test_phase0_config_schema_is_valid_json():
    from pathlib import Path

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "prototypes"
        / "phase0_research"
        / "config"
        / "quantum_input_vector.schema.json"
    )
    with open(schema_path) as f:
        schema = json.load(f)
    assert isinstance(schema, dict)


def test_no_secret_looking_content_in_research_prototypes():
    """Regression guard: none of these research-prototype paths should
    contain a credential-looking assignment. Complements the repo-wide sweep."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    dirs = [
        root / "prototypes" / "pqk" / "p0_1_qrbf_ocsvm_replication",
        root / "prototypes" / "phase0_research",
        root / "prototypes" / "phase1_research",
    ]
    pattern = re.compile(
        r"(token|api[_-]?key|secret|password)\s*=\s*['\"][A-Za-z0-9+/_=-]{12,}['\"]",
        re.IGNORECASE,
    )
    hits = []
    for d in dirs:
        for f in d.rglob("*.py"):
            text = f.read_text(errors="ignore")
            if pattern.search(text):
                hits.append(str(f))
    assert hits == [], f"secret-looking assignment found in: {hits}"
