"""Smoke tests for the PQK historical-prototype imports.

These preserve Gen1 research/experiment code as development history, not as
production algorithms (see algorithms/pqk/ for the maintained implementation).
The scripts here are not independently runnable (they need real market data
that isn't shipped in this repo) — these tests only confirm they still
import/parse cleanly, i.e. haven't silently bitrotted.
"""
import numpy as np


def test_original_kernel_phase0_imports_and_matches_expected_api():
    from prototypes.pqk._original_kernel_phase0 import (
        FEATURE_MAPS,
        train_fqk_svm,
        train_pqk_svm,
        AngleCNOTMap,
        ZZFeatureMap,
        CPMap,
    )

    assert set(FEATURE_MAPS) >= {"AngleCNOT", "ZZFeatureMap", "CPMap"}
    assert callable(train_fqk_svm)
    assert callable(train_pqk_svm)


def test_original_kernel_phase0_feature_map_runs():
    from prototypes.pqk._original_kernel_phase0 import AngleCNOTMap

    fmap = AngleCNOTMap()
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    state = fmap.encode(x) if hasattr(fmap, "encode") else fmap(x)
    assert state is not None


def test_phase1b_train_baselines_import_and_run():
    from prototypes.pqk.phase1b_train import LogisticBaseline, MajorityBaseline

    train_data = [{"features_8q": [0.1] * 8, "label": i % 2} for i in range(20)]
    val_data = [{"features_8q": [0.1] * 8, "label": i % 2} for i in range(10)]

    maj = MajorityBaseline(train_data)
    _, maj_acc = maj.evaluate(val_data)
    assert 0.0 <= maj_acc <= 1.0

    lr = LogisticBaseline(n_features=8)
    lr.train_epoch(train_data)
    _, lr_acc = lr.evaluate(val_data)
    assert 0.0 <= lr_acc <= 1.0


def test_phase2b_kernel_train_script_parses_and_imports():
    # Confirms the repointed imports (prototypes.pqk._original_kernel_phase0,
    # prototypes.pqk.phase1b_train) resolve correctly — the historical script
    # itself needs a real dataset file to actually run end to end, so this
    # test only proves it hasn't bitrotted at the import level.
    import importlib

    mod = importlib.import_module("prototypes.pqk.phase2b_kernel_train")
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_baselines")
