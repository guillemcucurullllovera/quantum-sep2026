"""Public-release guards for withdrawn, non-redistributable QRC claims."""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RETIRED_DIR = REPO_ROOT / "benchmarks/algorithms/qrc/_RETIRED_ADVANTAGE_CLAIMS"
RETIRED_SOURCE_PATHS = (
    "benchmarks/algorithms/qrc/_RETIRED_ADVANTAGE_CLAIMS/qrc_investor_demo.ipynb",
    "benchmarks/algorithms/qrc/_RETIRED_ADVANTAGE_CLAIMS/make_slides.py",
    "benchmarks/algorithms/qrc/_RETIRED_ADVANTAGE_CLAIMS/RETIRED.md",
)

# Source roots that must never IMPORT/reference a retired artifact from
# executable code. Deliberately excludes the retired directory itself, this
# test file, and docs/ — documentation (including this platform's own
# closeout/remediation reports) legitimately discusses the retirement by
# name; that is narrative, not a live code reference, and must not trip
# this check.
LIVE_SOURCE_ROOTS = [
    "core", "algorithms", "algorithms_extended", "applications", "hardware",
    "classical", "experiments", "integrations", "prototypes", "tools",
    "examples",
]
LIVE_BENCHMARKS_EXCLUDING_RETIRED = [
    p for p in (REPO_ROOT / "benchmarks").rglob("*")
    if p.is_file() and RETIRED_DIR not in p.parents
]

# Retired filenames that must never be described as active/runnable in a
# live (non-retired) Markdown doc under benchmarks/ — the exact gap an
# independent adversarial review found: benchmarks/algorithms/qrc/README.md
# still described make_slides.py at its old path as "runnable as-is".
RETIRED_FILENAMES = ("make_slides.py", "qrc_investor_demo.ipynb")


def test_retired_claim_artifacts_are_excluded_with_provenance():
    """Do not republish withdrawn investor claims to keep a source-era test green."""
    report = json.loads((REPO_ROOT / "SOURCE_COVERAGE.json").read_text())
    entries = {row["path"]: row for row in report["source_inventory"]}
    for source_path in RETIRED_SOURCE_PATHS:
        assert source_path in entries
        assert entries[source_path]["disposition"] == "EXCLUDED"
        assert "retired" in entries[source_path]["reason"].lower() or "historical" in entries[source_path]["reason"].lower()
        assert entries[source_path]["destination"] is None


def test_no_live_source_imports_retired_artifacts():
    pattern = re.compile(r"_RETIRED_ADVANTAGE_CLAIMS")
    offenders = []
    for root_name in LIVE_SOURCE_ROOTS:
        root = REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if pattern.search(path.read_text(errors="ignore")):
                offenders.append(str(path))
    for path in LIVE_BENCHMARKS_EXCLUDING_RETIRED:
        if path.suffix == ".py" and pattern.search(path.read_text(errors="ignore")):
            offenders.append(str(path))
    assert not offenders, f"live source must not reference retired artifacts: {offenders}"


def test_no_live_markdown_describes_retired_files_as_active():
    """Extends the guard suite beyond .py files, per the adversarial
    review's finding that benchmarks/algorithms/qrc/README.md (a Markdown
    file, never scanned by test_no_live_source_imports_retired_artifacts)
    still described make_slides.py at its pre-retirement path as
    "runnable as-is". Any live Markdown doc under benchmarks/ that mentions
    a retired filename must also say RETIRED somewhere in the same file."""
    offenders = []
    for path in LIVE_BENCHMARKS_EXCLUDING_RETIRED:
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(errors="ignore")
        mentions_retired_file = any(name in text for name in RETIRED_FILENAMES)
        if mentions_retired_file and "RETIRED" not in text:
            offenders.append(str(path))
    assert not offenders, (
        f"live Markdown doc mentions a retired filename without noting its "
        f"retired status: {offenders}"
    )


def test_no_orphaned_quantum_honest_positioning_reference():
    """docs/QUANTUM_HONEST_POSITIONING.md was referenced but never existed
    (QUANTUM-0 finding). Every reference must point at a doc that exists."""
    pattern = re.compile(r"docs/QUANTUM_HONEST_POSITIONING\.md")
    offenders = []
    for root_name in LIVE_SOURCE_ROOTS:
        root = REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if pattern.search(path.read_text(errors="ignore")):
                offenders.append(str(path))
    assert not offenders, f"dangling doc reference still present: {offenders}"
