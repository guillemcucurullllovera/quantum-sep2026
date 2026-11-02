"""Smoke tests: every example in examples/ must run to completion without
error. Kept fast (small qubit counts / sample sizes already baked into
each example's main()); this is about "does it execute end-to-end",
not about validating numerical results (each example's underlying
algorithm/application already has its own real tests for that).
"""
import importlib

import pytest

_EXAMPLE_MODULES = [
    "examples.qaoa_portfolio_example",
    "examples.qae_option_pricing_example",
    "examples.qrc_time_series_example",
    "examples.pqk_classification_example",
    "examples.vqc_qnn_classification_example",
    "examples.hardware_backend_transpile_noise_example",
    "examples.experiment_config_driven_example",
]


@pytest.mark.parametrize("module_name", _EXAMPLE_MODULES)
def test_example_runs_end_to_end(module_name, capsys):
    module = importlib.import_module(module_name)
    module.main()
    out = capsys.readouterr().out
    assert len(out) > 0, f"{module_name}.main() produced no output"
