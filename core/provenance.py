"""Provenance stamping: every ExperimentResult should be able to answer
"what code, what environment, what upstream citation produced this?" without
relying on tribal knowledge or a README nobody reads at run time.
"""
from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def package_versions(packages: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def environment_fingerprint() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(
            ["numpy", "scipy", "qiskit", "qiskit-ibm-runtime", "qiskit-aer", "qiskit-algorithms"]
        ),
    }


def build_provenance_note(*, generation: str | None, adapted_from: str | None) -> str:
    """Human-readable provenance line, meant to be stored in
    ExperimentResult.notes or a run's metadata so results are self-describing
    when read out of the registry months later.
    """
    parts = []
    if generation:
        parts.append(f"generation={generation}")
    if adapted_from:
        parts.append(f"adapted_from={adapted_from}")
    return "; ".join(parts) if parts else "internally authored, no adaptation"
