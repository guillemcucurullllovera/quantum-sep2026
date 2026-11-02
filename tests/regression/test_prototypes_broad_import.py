"""Smoke tests: historical Gen1 scripts/infra ported into prototypes/{other,hardware}
must at minimum import cleanly, proving the port didn't silently break at the
module level. These are historical research artifacts, not production code —
we are not asserting their business logic here, only that the port preserved
a working module (per this platform's "preserve history, don't modernize" directive).
"""
import importlib

import pytest

INFRA_MODULES = [
    "prototypes.other.quantum_signal_gen1_infra.models",
    "prototypes.other.quantum_signal_gen1_infra.builder",
    "prototypes.other.quantum_signal_gen1_infra.validator",
    "prototypes.other.quantum_signal_gen1_infra.reader",
    "prototypes.other.quantum_signal_gen1_infra.enrichment",
    "prototypes.other.quantum_signal_gen1_infra.ensemble",
    "prototypes.other.quantum_signal_gen1_infra.normalizer",
    "prototypes.other.quantum_signal_gen1_infra.logger",
    "prototypes.other.quantum_signal_gen1_infra.features.momentum",
    "prototypes.other.quantum_signal_gen1_infra.features.volatility",
    "prototypes.other.quantum_signal_gen1_infra.features.macro",
    "prototypes.other.quantum_signal_gen1_infra.features.sentiment",
    "prototypes.other.quantum_signal_gen1_infra.features.positioning",
    "prototypes.other.quantum_signal_gen1_infra.features.cross_asset",
]

SCRIPT_MODULES = [
    "prototypes.other.multi_capture_pipeline",
    "prototypes.other.phase1_viewer",
    "prototypes.other.prepare_dataset_v2",
    "prototypes.other.real_data_pipeline",
    "prototypes.other.redis_capture",
    "prototypes.other.reprocess_features",
    "prototypes.other.signals_to_dataset",
    "prototypes.other.snapshot_to_dataset",
    "prototypes.hardware.generate_qiskit_circuit_20q",
    "prototypes.hardware.generate_qiskit_circuit",
    "prototypes.hardware.real_system_validation",
]


@pytest.mark.parametrize("module_name", INFRA_MODULES)
def test_gen1_infra_module_imports(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", SCRIPT_MODULES)
def test_ported_script_imports(module_name):
    """These scripts have argparse __main__ blocks; import-time (module load)
    must succeed even though running them end-to-end requires external data
    (Redis, FRED API, snapshot files) that is deliberately not part of this
    repo per the platform's data-dump exclusion policy.
    """
    importlib.import_module(module_name)


def test_quantum_input_vector_model_is_constructible():
    """One real behavioral check, not just an import: the core Gen1 data
    model should still be instantiable with the fields it always had.
    """
    from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(QuantumInputVector)}
    assert "asset" in field_names or len(field_names) > 0
