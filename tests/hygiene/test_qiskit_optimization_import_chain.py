"""Regression test for a specific, previously-false claim: QUANTUM-1A's
original closeout asserted the direct (non-vendored) `qiskit_optimization`
import sites had "zero production consumers." An independent adversarial
review proved this false — importing the production portfolio training
module transitively, eagerly imports `qiskit_optimization` via
`upstream.training.tqa_trainer -> upstream.evaluation.EVALUATORS ->
efficient_depth_one/light_cone -> upstream.utils.graph_utils`.

This test pins that finding down mechanically so a future contributor
cannot silently drift back onto the false "test/utility-only" assumption
while deciding whether it's safe to let this dependency break.

See docs/quantum/QUANTUM1A_REVIEW_REMEDIATION.md, finding P0-1.
"""
import subprocess
import sys


def test_production_portfolio_module_imports_qiskit_optimization_transitively():
    """Runs in a fresh subprocess (not the test-runner's own interpreter,
    which may have already imported qiskit_optimization via an unrelated
    test) so the sys.modules check is unambiguous."""
    script = (
        "import sys\n"
        "assert 'qiskit_optimization' not in sys.modules\n"
        "import applications.portfolio.training_pipeline_backend\n"
        "assert 'qiskit_optimization' in sys.modules, "
        "'expected qiskit_optimization to land in sys.modules via the "
        "training_pipeline_backend -> tqa_trainer -> evaluation -> "
        "graph_utils chain'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_graph_utils_module_lands_in_sys_modules_via_production_chain():
    """Narrower than the test above: confirms the specific module the
    review traced the dependency to, not just the package as a whole."""
    script = (
        "import sys\n"
        "import applications.portfolio.training_pipeline_backend\n"
        "assert any('upstream.utils.graph_utils' in m for m in sys.modules), "
        "'expected upstream.utils.graph_utils in sys.modules'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_data_utils_and_problem_classes_are_not_reached_by_production_chain():
    """The other two of the five risky-import files ARE genuinely not
    reached by the production chain — confirmed dynamically, not assumed.
    If this ever flips to True, the corresponding source comments and
    docs/quantum/QUANTUM1A_REVIEW_REMEDIATION.md must be updated to match,
    not left stale."""
    script = (
        "import sys\n"
        "import applications.portfolio.training_pipeline_backend\n"
        "utils_mods = [m for m in sys.modules "
        "if 'qaoa_training_pipeline_asymmetrica.upstream.utils' in m]\n"
        "assert not any('data_utils' in m for m in utils_mods), utils_mods\n"
        "assert not any('problem_classes' in m for m in utils_mods), utils_mods\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
