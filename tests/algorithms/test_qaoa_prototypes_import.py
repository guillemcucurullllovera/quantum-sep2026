"""Historical Gen1 QAOA prototypes (prototypes/qaoa/) aren't unit-testable
in the normal sense — they're preserved research/sweep scripts that depend
on live market data for a full run. This test asserts the one thing that
matters for "preserved, not bit-rotted": every script still imports cleanly
against this repo's current module layout.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PROTOTYPES_QAOA = Path(__file__).resolve().parents[2] / "prototypes" / "qaoa"

SCRIPTS = [
    "qaoa_backtest.py",
    "phase2d_qaoa_optimize.py",
    "ablation_fast.py",
    "n15_overnight.py",
    "n20_overnight.py",
    "overnight_sweep.py",
]


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_prototype_script_imports_cleanly(script_name):
    path = PROTOTYPES_QAOA / script_name
    assert path.exists(), f"expected historical prototype at {path}"
    module_name = f"_qaoa_prototype_{script_name.replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)


def test_base_scripts_expose_quantum_classes():
    """The two scripts that actually run QAOA must resolve the real classes,
    not silently fall back to something else."""
    for script_name in ("qaoa_backtest.py", "phase2d_qaoa_optimize.py"):
        path = PROTOTYPES_QAOA / script_name
        module_name = f"_qaoa_prototype_check_{script_name.replace('.py', '')}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "PortfolioQUBO")
            assert hasattr(module, "QAOACircuit")
            assert hasattr(module, "QAOAOptimizer")
        finally:
            sys.modules.pop(module_name, None)
