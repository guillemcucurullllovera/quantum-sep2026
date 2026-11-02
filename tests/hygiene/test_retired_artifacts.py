"""QUANTUM-1A Phase 0: retirement/quarantine guards for withdrawn
advantage-claim artifacts (see benchmarks/algorithms/qrc/_RETIRED_ADVANTAGE_CLAIMS/RETIRED.md).

These tests fail the suite if a retired artifact is deleted (evidence
must be preserved, not erased), if its reuse guard is removed, or if any
live source path starts importing/referencing it again.
"""
import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RETIRED_DIR = REPO_ROOT / "benchmarks/algorithms/qrc/_RETIRED_ADVANTAGE_CLAIMS"

# The original notebook's sha256 at base commit 799198c1f1695619a9cdbe4c998b5fe7b7dcf8d0
# ("Initial quantum platform release") — see RETIRED.md's "Immutability record".
# The retired notebook's bytes must never drift from this: no banner, no
# metadata, nothing inserted into the historical artifact itself.
ORIGINAL_NOTEBOOK_SHA256 = (
    "3fce18b37df70882216257812c593b761cdcb2b0933bd1caf360bf6a8dd562fb"
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


def test_retired_directory_and_marker_exist():
    assert RETIRED_DIR.is_dir(), "retirement quarantine directory must exist"
    marker = RETIRED_DIR / "RETIRED.md"
    assert marker.is_file()
    text = marker.read_text()
    for required in ("RETIRED", "INVALIDATED", "NOT FOR EXTERNAL USE", "Q10"):
        assert required in text


def test_retired_artifacts_preserved_not_deleted():
    notebook = RETIRED_DIR / "qrc_investor_demo.ipynb"
    slides = RETIRED_DIR / "make_slides.py"
    assert notebook.is_file(), "historical evidence must be preserved, not deleted"
    assert slides.is_file(), "historical evidence must be preserved, not deleted"


def test_retired_notebook_is_byte_identical_to_original():
    """The historical artifact's bytes must never change — no banner, no
    metadata, nothing inserted. Retirement state lives in RETIRED.md and
    the runtime guard below, never in the artifact itself. See RETIRED.md's
    "Immutability record" and docs/quantum/QUANTUM1A_REVIEW_REMEDIATION.md,
    finding P1-2."""
    notebook = RETIRED_DIR / "qrc_investor_demo.ipynb"
    digest = hashlib.sha256(notebook.read_bytes()).hexdigest()
    assert digest == ORIGINAL_NOTEBOOK_SHA256, (
        f"qrc_investor_demo.ipynb has drifted from its recorded original "
        f"hash ({digest} != {ORIGINAL_NOTEBOOK_SHA256}) — the historical "
        "artifact must remain byte-identical; retirement state belongs in "
        "RETIRED.md, not in the notebook's own bytes."
    )


def test_retired_md_documents_the_immutability_hash():
    text = (RETIRED_DIR / "RETIRED.md").read_text()
    assert ORIGINAL_NOTEBOOK_SHA256 in text


def test_make_slides_raises_on_execution():
    text = (RETIRED_DIR / "make_slides.py").read_text()
    assert "raise RuntimeError" in text
    assert "RETIRED" in text.split("raise RuntimeError")[1][:400]


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
