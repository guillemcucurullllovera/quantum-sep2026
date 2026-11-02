"""Financial classification applications composed from one or more
algorithm modules.

This module adds a unified product layer on top of the comparison
scripts (`three_way_comparison.py`, `kernel_classifier.py`,
`vqc_classifier.py`, `qnn_regression.py`):

  task_spec.py            -- ClassificationTaskSpec / Algorithm / TaskType
  dispatcher.py            -- dispatch(spec, X_train, y_train, X_test, y_test)
                               routes to any of ~16 concrete implementations
  result.py                -- DispatchResult, the shared output schema
  errors.py                -- UnsupportedTaskError / AlgorithmUnavailableError / LeakageError
  metrics_ext.py            -- precision/recall/F1/AUC (dependency-light)
  preprocessing.py          -- Preprocessor (fracdiff + train-only standardization)
  rolling_evaluation.py     -- run_rolling_evaluation: metric trajectory across windows
  calibration_utils.py      -- calibrate_dispatch: held-out Platt scaling for ANY algorithm
  benchmark_report.py       -- run_comprehensive_comparison: every applicable method, one task
  cli.py                    -- python -m applications.classification.cli ...

See `dispatcher.py`'s module docstring for the full algorithm routing table.
"""
