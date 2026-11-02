# Complete source coverage

1969 source artifacts retained: 715 first-party-classified files (482 Python modules/runners outside the top-level tests tree), plus 1254 separately attributed upstream/mixed-permissive files. Mixed packages also contain first-party finance adaptations, which are deliberately retained rather than discarded merely because of their directory name. No authorship claim is made over upstream code.

## Source branches

| Branch | Commit |
|---|---|
| `main` | `799198c1f1695619a9cdbe4c998b5fe7b7dcf8d0` |
| `quantum/quantum1b-research-20260930` | `ab1881eaa082312bba73399fca4c8f8aca42b0a7` |
| `checkpoint/quantum1a-accepted-20260930` | `b7ff60c9725dc9b8e2689aec458dc64a79c667a7` |

## Architecture coverage

| Root | Retained files |
|---|---|
| `LICENSES` | 14 |
| `THIRD_PARTY_NOTICES.md` | 1 |
| `algorithms` | 66 |
| `algorithms_extended` | 1174 |
| `applications` | 102 |
| `benchmarks` | 29 |
| `classical` | 32 |
| `core` | 20 |
| `examples` | 10 |
| `experiments` | 70 |
| `hardware` | 78 |
| `integrations` | 2 |
| `notebook_sources` | 6 |
| `prototypes` | 68 |
| `tests` | 288 |
| `tools` | 9 |

## Families included

- QAE/IQAE/MLAE and noise-aware likelihood
- Gen1/Gen2 QAOA, QUBO, XY/Dicke mixers, CVaR QAOA
- Projected/fidelity quantum kernels, Nyström and alignment
- VQC/QNN v1-v5 and layerwise training
- Quantum reservoirs and ESN/HMM/classical comparisons
- Readout mitigation, ZNE, PEC and dynamical decoupling
- Finance option/barrier pricing, Greeks, portfolio constraints, tail risk, regime and anomaly research
- Exact MILP/brute-force and heuristic competitor gates
- Hardware resource/cost budgets, frozen controls, lineage, authorization and evidence gates

## Missing or unvalidated

- No identified first-party Python model family intentionally omitted; live QPU capability and historical empirical claims remain unvalidated

## Exclusions and license boundary

SOURCE_COVERAGE.json enumerates every file in each inspected branch, exact exclusion reasons and superseded versions, all retained destinations/hashes, available tests, experiment runners, dependencies and unreproduced empirical claims. Original reports, hardware/result archives, environment examples and retired investor presentations are excluded. Notebook outputs and metadata are removed; code-only archival extracts are retained separately and are not validated runnable experiments. No scientific Python implementation is excluded for unavailable private data. The only excluded Python files are presentation/document-build drivers and retired slides; their source estate is preserved privately, not erased.

THIRD_PARTY_NOTICES.md and LICENSES/ preserve each permitted upstream license, including Apache-2.0/MIT attribution. Root first-party licensing does not relicense those components. Provider observations, proprietary solvers and restricted source are not redistributed.

## First-party Python mapping

| Source | Destination | Branch |
|---|---|
| `algorithms/__init__.py` | `algorithms/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/__init__.py` | `algorithms/pqk/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/caching.py` | `algorithms/pqk/caching.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/feature_maps.py` | `algorithms/pqk/feature_maps.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/kernels.py` | `algorithms/pqk/kernels.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/kernels_gen1.py` | `algorithms/pqk/kernels_gen1.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/kta_alignment.py` | `algorithms/pqk/kta_alignment.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/noise_study.py` | `algorithms/pqk/noise_study.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/nystrom.py` | `algorithms/pqk/nystrom.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/pqk_v2.py` | `algorithms/pqk/pqk_v2.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/pqk/training.py` | `algorithms/pqk/training.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/__init__.py` | `algorithms/qae_iqae/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/backends.py` | `algorithms/qae_iqae/backends.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/barrier_circuits.py` | `algorithms/qae_iqae/barrier_circuits.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/deliverables/__init__.py` | `algorithms/qae_iqae/deliverables/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/erosion_audit.py` | `algorithms/qae_iqae/erosion_audit.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/finance_integrands.py` | `algorithms/qae_iqae/finance_integrands.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/hardware/__init__.py` | `algorithms/qae_iqae/hardware/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/hardware/noise_aware.py` | `algorithms/qae_iqae/hardware/noise_aware.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/iqae.py` | `algorithms/qae_iqae/iqae.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/mitigation.py` | `algorithms/qae_iqae/mitigation.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qae_iqae/register.py` | `algorithms/qae_iqae/register.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/__init__.py` | `algorithms/qaoa_qubo/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/classical_baseline.py` | `algorithms/qaoa_qubo/classical_baseline.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/gen1/__init__.py` | `algorithms/qaoa_qubo/gen1/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/gen1/qaoa_portfolio.py` | `algorithms/qaoa_qubo/gen1/qaoa_portfolio.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/gen2/__init__.py` | `algorithms/qaoa_qubo/gen2/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/gen2/mixers.py` | `algorithms/qaoa_qubo/gen2/mixers.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/gen2/qaoa_v2.py` | `algorithms/qaoa_qubo/gen2/qaoa_v2.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/gen2/sk_warmstart.py` | `algorithms/qaoa_qubo/gen2/sk_warmstart.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/qubo_builder.py` | `algorithms/qaoa_qubo/qubo_builder.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qaoa_qubo/solvers.py` | `algorithms/qaoa_qubo/solvers.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/__init__.py` | `algorithms/qrc/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/encoders.py` | `algorithms/qrc/encoders.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/gen2_v1_original/__init__.py` | `algorithms/qrc/gen2_v1_original/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/gen2_v1_original/qrc_encoder.py` | `algorithms/qrc/gen2_v1_original/qrc_encoder.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/gen2_v1_original/qrc_v1.py` | `algorithms/qrc/gen2_v1_original/qrc_v1.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/gen2_v1_original/tests/__init__.py` | `algorithms/qrc/gen2_v1_original/tests/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/gen2_v1_original/tests/test_qrc_v1.py` | `algorithms/qrc/gen2_v1_original/tests/test_qrc_v1.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/reservoir.py` | `algorithms/qrc/reservoir.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/reservoir_math.py` | `algorithms/qrc/reservoir_math.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/qrc/training.py` | `algorithms/qrc/training.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/__init__.py` | `algorithms/vqc_qml/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/ansatz.py` | `algorithms/vqc_qml/ansatz.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/calibration.py` | `algorithms/vqc_qml/calibration.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/encoding.py` | `algorithms/vqc_qml/encoding.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/training.py` | `algorithms/vqc_qml/training.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v1/__init__.py` | `algorithms/vqc_qml/v1/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v1/circuit.py` | `algorithms/vqc_qml/v1/circuit.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v2/__init__.py` | `algorithms/vqc_qml/v2/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v2/circuit_v2.py` | `algorithms/vqc_qml/v2/circuit_v2.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v3/__init__.py` | `algorithms/vqc_qml/v3/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v3/circuit_v3.py` | `algorithms/vqc_qml/v3/circuit_v3.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v4/__init__.py` | `algorithms/vqc_qml/v4/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v4/circuit_v4.py` | `algorithms/vqc_qml/v4/circuit_v4.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v4/circuit_v4_qiskit.py` | `algorithms/vqc_qml/v4/circuit_v4_qiskit.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v5/__init__.py` | `algorithms/vqc_qml/v5/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v5/ansatz.py` | `algorithms/vqc_qml/v5/ansatz.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v5/layerwise_trainer.py` | `algorithms/vqc_qml/v5/layerwise_trainer.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v5/tests/__init__.py` | `algorithms/vqc_qml/v5/tests/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v5/tests/test_gradient_variance.py` | `algorithms/vqc_qml/v5/tests/test_gradient_variance.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v5/tests/test_vqc_v5.py` | `algorithms/vqc_qml/v5/tests/test_vqc_v5.py` | `quantum/quantum1b-research-20260930` |
| `algorithms/vqc_qml/v5/vqc_v5.py` | `algorithms/vqc_qml/v5/vqc_v5.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/__init__.py` | `algorithms_extended/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/error_mitigation/zne.py` | `algorithms_extended/error_mitigation/zne.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/qaoa_initialization/__init__.py` | `algorithms_extended/qaoa_initialization/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/qaoa_initialization/interp_strategy.py` | `algorithms_extended/qaoa_initialization/interp_strategy.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/qaoa_initialization/linear_ramp.py` | `algorithms_extended/qaoa_initialization/linear_ramp.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/qaoa_initialization/strategies.py` | `algorithms_extended/qaoa_initialization/strategies.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/qrbf_ocsvm_anomaly/__init__.py` | `algorithms_extended/qrbf_ocsvm_anomaly/__init__.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/qrbf_ocsvm_anomaly/hardware_variant.py` | `algorithms_extended/qrbf_ocsvm_anomaly/hardware_variant.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/qrbf_ocsvm_anomaly/kernels.py` | `algorithms_extended/qrbf_ocsvm_anomaly/kernels.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/qrbf_ocsvm_anomaly/replication.py` | `algorithms_extended/qrbf_ocsvm_anomaly/replication.py` | `quantum/quantum1b-research-20260930` |
| `algorithms_extended/qrbf_ocsvm_anomaly/sparkov_data.py` | `algorithms_extended/qrbf_ocsvm_anomaly/sparkov_data.py` | `quantum/quantum1b-research-20260930` |
| `applications/__init__.py` | `applications/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/anomaly_detection/__init__.py` | `applications/anomaly_detection/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/anomaly_detection/fraud_detector.py` | `applications/anomaly_detection/fraud_detector.py` | `quantum/quantum1b-research-20260930` |
| `applications/anomaly_detection/general_anomaly.py` | `applications/anomaly_detection/general_anomaly.py` | `quantum/quantum1b-research-20260930` |
| `applications/anomaly_detection/synthetic_trade_anomalies.py` | `applications/anomaly_detection/synthetic_trade_anomalies.py` | `quantum/quantum1b-research-20260930` |
| `applications/anomaly_detection/synthetic_transactions.py` | `applications/anomaly_detection/synthetic_transactions.py` | `quantum/quantum1b-research-20260930` |
| `applications/asset_allocation/__init__.py` | `applications/asset_allocation/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/asset_allocation/allocator.py` | `applications/asset_allocation/allocator.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/__init__.py` | `applications/classification/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/benchmark_report.py` | `applications/classification/benchmark_report.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/calibration_utils.py` | `applications/classification/calibration_utils.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/cli.py` | `applications/classification/cli.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/dispatcher.py` | `applications/classification/dispatcher.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/errors.py` | `applications/classification/errors.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/kernel_classifier.py` | `applications/classification/kernel_classifier.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/metrics_ext.py` | `applications/classification/metrics_ext.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/preprocessing.py` | `applications/classification/preprocessing.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/qnn_regression.py` | `applications/classification/qnn_regression.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/result.py` | `applications/classification/result.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/rolling_evaluation.py` | `applications/classification/rolling_evaluation.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/synthetic_financial_dataset.py` | `applications/classification/synthetic_financial_dataset.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/synthetic_volatility_dataset.py` | `applications/classification/synthetic_volatility_dataset.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/task_spec.py` | `applications/classification/task_spec.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/three_way_comparison.py` | `applications/classification/three_way_comparison.py` | `quantum/quantum1b-research-20260930` |
| `applications/classification/vqc_classifier.py` | `applications/classification/vqc_classifier.py` | `quantum/quantum1b-research-20260930` |
| `applications/derivatives/__init__.py` | `applications/derivatives/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/derivatives/portfolio_hedging.py` | `applications/derivatives/portfolio_hedging.py` | `quantum/quantum1b-research-20260930` |
| `applications/derivatives/scenario_analysis.py` | `applications/derivatives/scenario_analysis.py` | `quantum/quantum1b-research-20260930` |
| `applications/derivatives/scenario_grid.py` | `applications/derivatives/scenario_grid.py` | `quantum/quantum1b-research-20260930` |
| `applications/hedging/__init__.py` | `applications/hedging/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/hedging/options_based_hedge.py` | `applications/hedging/options_based_hedge.py` | `quantum/quantum1b-research-20260930` |
| `applications/hedging/regime_conditional_hedge.py` | `applications/hedging/regime_conditional_hedge.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/__init__.py` | `applications/options/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/barrier_pricer.py` | `applications/options/barrier_pricer.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/cli.py` | `applications/options/cli.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/comparison_report.py` | `applications/options/comparison_report.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/domain.py` | `applications/options/domain.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/execution.py` | `applications/options/execution.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/greeks.py` | `applications/options/greeks.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/iqae/__init__.py` | `applications/options/iqae/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/monte_carlo/__init__.py` | `applications/options/monte_carlo/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/option_pricer.py` | `applications/options/option_pricer.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/pricing_methods.py` | `applications/options/pricing_methods.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/qae/__init__.py` | `applications/options/qae/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/options/resource_and_convergence.py` | `applications/options/resource_and_convergence.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/__init__.py` | `applications/portfolio/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/benchmark_suite.py` | `applications/portfolio/benchmark_suite.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/classical/__init__.py` | `applications/portfolio/classical/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/classical_backend.py` | `applications/portfolio/classical_backend.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/cli.py` | `applications/portfolio/cli.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/data_pipeline.py` | `applications/portfolio/data_pipeline.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/domain.py` | `applications/portfolio/domain.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/exceptions.py` | `applications/portfolio/exceptions.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/gen1_backend.py` | `applications/portfolio/gen1_backend.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/hardware_exec.py` | `applications/portfolio/hardware_exec.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/multi_solver.py` | `applications/portfolio/multi_solver.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/optimizer.py` | `applications/portfolio/optimizer.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/prototypes/__init__.py` | `applications/portfolio/prototypes/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/qaoa/__init__.py` | `applications/portfolio/qaoa/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/qp_backend.py` | `applications/portfolio/qp_backend.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/report.py` | `applications/portfolio/report.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/scenario.py` | `applications/portfolio/scenario.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/synthetic_market.py` | `applications/portfolio/synthetic_market.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/training_pipeline_backend.py` | `applications/portfolio/training_pipeline_backend.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/validation.py` | `applications/portfolio/validation.py` | `quantum/quantum1b-research-20260930` |
| `applications/portfolio/xy_mixer_backend.py` | `applications/portfolio/xy_mixer_backend.py` | `quantum/quantum1b-research-20260930` |
| `applications/regime_detection/__init__.py` | `applications/regime_detection/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/regime_detection/hmm_baseline_comparison.py` | `applications/regime_detection/hmm_baseline_comparison.py` | `quantum/quantum1b-research-20260930` |
| `applications/regime_detection/regime_classifier.py` | `applications/regime_detection/regime_classifier.py` | `quantum/quantum1b-research-20260930` |
| `applications/regime_detection/regime_product.py` | `applications/regime_detection/regime_product.py` | `quantum/quantum1b-research-20260930` |
| `applications/regime_detection/synthetic_regime_dataset.py` | `applications/regime_detection/synthetic_regime_dataset.py` | `quantum/quantum1b-research-20260930` |
| `applications/scenario_analysis/__init__.py` | `applications/scenario_analysis/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/scenario_analysis/domain.py` | `applications/scenario_analysis/domain.py` | `quantum/quantum1b-research-20260930` |
| `applications/scenario_analysis/generator.py` | `applications/scenario_analysis/generator.py` | `quantum/quantum1b-research-20260930` |
| `applications/scenario_analysis/options_adapter.py` | `applications/scenario_analysis/options_adapter.py` | `quantum/quantum1b-research-20260930` |
| `applications/scenario_analysis/runner.py` | `applications/scenario_analysis/runner.py` | `quantum/quantum1b-research-20260930` |
| `applications/scenario_analysis/tail_risk_adapter.py` | `applications/scenario_analysis/tail_risk_adapter.py` | `quantum/quantum1b-research-20260930` |
| `applications/tail_risk/__init__.py` | `applications/tail_risk/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/tail_risk/backtest.py` | `applications/tail_risk/backtest.py` | `quantum/quantum1b-research-20260930` |
| `applications/tail_risk/cli.py` | `applications/tail_risk/cli.py` | `quantum/quantum1b-research-20260930` |
| `applications/tail_risk/comparison.py` | `applications/tail_risk/comparison.py` | `quantum/quantum1b-research-20260930` |
| `applications/tail_risk/domain.py` | `applications/tail_risk/domain.py` | `quantum/quantum1b-research-20260930` |
| `applications/tail_risk/errors.py` | `applications/tail_risk/errors.py` | `quantum/quantum1b-research-20260930` |
| `applications/tail_risk/integration.py` | `applications/tail_risk/integration.py` | `quantum/quantum1b-research-20260930` |
| `applications/tail_risk/var_cvar.py` | `applications/tail_risk/var_cvar.py` | `quantum/quantum1b-research-20260930` |
| `applications/tail_risk/walk_forward.py` | `applications/tail_risk/walk_forward.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/__init__.py` | `applications/time_series/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/cli.py` | `applications/time_series/cli.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/dispatcher.py` | `applications/time_series/dispatcher.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/domain.py` | `applications/time_series/domain.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/errors.py` | `applications/time_series/errors.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/evaluation.py` | `applications/time_series/evaluation.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/forecaster.py` | `applications/time_series/forecaster.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/methods.py` | `applications/time_series/methods.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/noise.py` | `applications/time_series/noise.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/preprocessing.py` | `applications/time_series/preprocessing.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/report.py` | `applications/time_series/report.py` | `quantum/quantum1b-research-20260930` |
| `applications/time_series/synthetic.py` | `applications/time_series/synthetic.py` | `quantum/quantum1b-research-20260930` |
| `applications/volatility/__init__.py` | `applications/volatility/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/volatility/kernels/__init__.py` | `applications/volatility/kernels/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/volatility/qrc/__init__.py` | `applications/volatility/qrc/__init__.py` | `quantum/quantum1b-research-20260930` |
| `applications/volatility/realized_volatility.py` | `applications/volatility/realized_volatility.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/__init__.py` | `benchmarks/__init__.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/__init__.py` | `benchmarks/algorithms/__init__.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/__init__.py` | `benchmarks/algorithms/qrc/__init__.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/compute_sharpe.py` | `benchmarks/algorithms/qrc/compute_sharpe.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/compute_sharpe_v3.py` | `benchmarks/algorithms/qrc/compute_sharpe_v3.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/qrc_1h_bench_v5.py` | `benchmarks/algorithms/qrc/qrc_1h_bench_v5.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/qrc_1h_benchmark.py` | `benchmarks/algorithms/qrc/qrc_1h_benchmark.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/qrc_1h_qr2.py` | `benchmarks/algorithms/qrc/qrc_1h_qr2.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/qrc_1h_v2.py` | `benchmarks/algorithms/qrc/qrc_1h_v2.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/qrc_1h_v2_fast.py` | `benchmarks/algorithms/qrc/qrc_1h_v2_fast.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/qrc_1h_v3.py` | `benchmarks/algorithms/qrc/qrc_1h_v3.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/qrc_m15_5d_benchmark.py` | `benchmarks/algorithms/qrc/qrc_m15_5d_benchmark.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/qrc_m15_v2.py` | `benchmarks/algorithms/qrc/qrc_m15_v2.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/algorithms/qrc/qrc_multifreq_benchmark.py` | `benchmarks/algorithms/qrc/qrc_multifreq_benchmark.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/applications/__init__.py` | `benchmarks/applications/__init__.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/mitigation/__init__.py` | `benchmarks/mitigation/__init__.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/noise/__init__.py` | `benchmarks/noise/__init__.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/quantum_vs_classical/__init__.py` | `benchmarks/quantum_vs_classical/__init__.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/quantum_vs_classical/anomaly.py` | `benchmarks/quantum_vs_classical/anomaly.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/quantum_vs_classical/pqk.py` | `benchmarks/quantum_vs_classical/pqk.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/quantum_vs_classical/qaoa.py` | `benchmarks/quantum_vs_classical/qaoa.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/quantum_vs_classical/qrc.py` | `benchmarks/quantum_vs_classical/qrc.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/report.py` | `benchmarks/report.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/reproducibility.py` | `benchmarks/reproducibility.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/resource_estimation/__init__.py` | `benchmarks/resource_estimation/__init__.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/runner.py` | `benchmarks/runner.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/scaling/__init__.py` | `benchmarks/scaling/__init__.py` | `quantum/quantum1b-research-20260930` |
| `benchmarks/simulator_vs_qpu/__init__.py` | `benchmarks/simulator_vs_qpu/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/__init__.py` | `classical/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/kernels/__init__.py` | `classical/kernels/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/kernels/baselines.py` | `classical/kernels/baselines.py` | `quantum/quantum1b-research-20260930` |
| `classical/machine_learning/__init__.py` | `classical/machine_learning/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/machine_learning/baselines.py` | `classical/machine_learning/baselines.py` | `quantum/quantum1b-research-20260930` |
| `classical/meta_agent/__init__.py` | `classical/meta_agent/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/meta_agent/monte_carlo_cvar.py` | `classical/meta_agent/monte_carlo_cvar.py` | `quantum/quantum1b-research-20260930` |
| `classical/meta_agent/ocsvm_anomaly.py` | `classical/meta_agent/ocsvm_anomaly.py` | `quantum/quantum1b-research-20260930` |
| `classical/meta_agent/regime_hmm.py` | `classical/meta_agent/regime_hmm.py` | `quantum/quantum1b-research-20260930` |
| `classical/meta_agent/tests/__init__.py` | `classical/meta_agent/tests/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/meta_agent/tests/test_monte_carlo_cvar.py` | `classical/meta_agent/tests/test_monte_carlo_cvar.py` | `quantum/quantum1b-research-20260930` |
| `classical/meta_agent/tests/test_ocsvm_anomaly.py` | `classical/meta_agent/tests/test_ocsvm_anomaly.py` | `quantum/quantum1b-research-20260930` |
| `classical/meta_agent/tests/test_regime_hmm.py` | `classical/meta_agent/tests/test_regime_hmm.py` | `quantum/quantum1b-research-20260930` |
| `classical/meta_agent/tests/test_regime_hmm_vs_quantum_comparator.py` | `classical/meta_agent/tests/test_regime_hmm_vs_quantum_comparator.py` | `quantum/quantum1b-research-20260930` |
| `classical/monte_carlo/__init__.py` | `classical/monte_carlo/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/monte_carlo/gbm.py` | `classical/monte_carlo/gbm.py` | `quantum/quantum1b-research-20260930` |
| `classical/optimization/__init__.py` | `classical/optimization/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/optimization/brute_force.py` | `classical/optimization/brute_force.py` | `quantum/quantum1b-research-20260930` |
| `classical/optimization/milp_solver.py` | `classical/optimization/milp_solver.py` | `quantum/quantum1b-research-20260930` |
| `classical/optimization/simulated_annealing.py` | `classical/optimization/simulated_annealing.py` | `quantum/quantum1b-research-20260930` |
| `classical/option_pricing/__init__.py` | `classical/option_pricing/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/option_pricing/black_scholes.py` | `classical/option_pricing/black_scholes.py` | `quantum/quantum1b-research-20260930` |
| `classical/option_pricing/monte_carlo_pricer.py` | `classical/option_pricing/monte_carlo_pricer.py` | `quantum/quantum1b-research-20260930` |
| `classical/portfolio/__init__.py` | `classical/portfolio/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/portfolio/markowitz.py` | `classical/portfolio/markowitz.py` | `quantum/quantum1b-research-20260930` |
| `classical/reservoir/__init__.py` | `classical/reservoir/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/reservoir/esn.py` | `classical/reservoir/esn.py` | `quantum/quantum1b-research-20260930` |
| `classical/risk/__init__.py` | `classical/risk/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/risk/parametric.py` | `classical/risk/parametric.py` | `quantum/quantum1b-research-20260930` |
| `classical/time_series/__init__.py` | `classical/time_series/__init__.py` | `quantum/quantum1b-research-20260930` |
| `classical/time_series/classical_baselines.py` | `classical/time_series/classical_baselines.py` | `quantum/quantum1b-research-20260930` |
| `classical/time_series/mamba_ssm_baseline.py` | `classical/time_series/mamba_ssm_baseline.py` | `quantum/quantum1b-research-20260930` |
| `core/__init__.py` | `core/__init__.py` | `quantum/quantum1b-research-20260930` |
| `core/cli.py` | `core/cli.py` | `quantum/quantum1b-research-20260930` |
| `core/config.py` | `core/config.py` | `quantum/quantum1b-research-20260930` |
| `core/dataset_interfaces.py` | `core/dataset_interfaces.py` | `quantum/quantum1b-research-20260930` |
| `core/errors.py` | `core/errors.py` | `quantum/quantum1b-research-20260930` |
| `core/evidence.py` | `core/evidence.py` | `quantum/quantum1b-research-20260930` |
| `core/experiment_registry.py` | `core/experiment_registry.py` | `quantum/quantum1b-research-20260930` |
| `core/feature_engineering/__init__.py` | `core/feature_engineering/__init__.py` | `quantum/quantum1b-research-20260930` |
| `core/feature_engineering/fracdiff.py` | `core/feature_engineering/fracdiff.py` | `quantum/quantum1b-research-20260930` |
| `core/feature_engineering/purged_cv.py` | `core/feature_engineering/purged_cv.py` | `quantum/quantum1b-research-20260930` |
| `core/feature_engineering/triple_barrier.py` | `core/feature_engineering/triple_barrier.py` | `quantum/quantum1b-research-20260930` |
| `core/logging_utils.py` | `core/logging_utils.py` | `quantum/quantum1b-research-20260930` |
| `core/metrics.py` | `core/metrics.py` | `quantum/quantum1b-research-20260930` |
| `core/provenance.py` | `core/provenance.py` | `quantum/quantum1b-research-20260930` |
| `core/registry.py` | `core/registry.py` | `quantum/quantum1b-research-20260930` |
| `core/result_store.py` | `core/result_store.py` | `quantum/quantum1b-research-20260930` |
| `core/schemas.py` | `core/schemas.py` | `quantum/quantum1b-research-20260930` |
| `core/seeds.py` | `core/seeds.py` | `quantum/quantum1b-research-20260930` |
| `core/serialization.py` | `core/serialization.py` | `quantum/quantum1b-research-20260930` |
| `core/verification.py` | `core/verification.py` | `quantum/quantum1b-research-20260930` |
| `examples/__init__.py` | `examples/__init__.py` | `quantum/quantum1b-research-20260930` |
| `examples/experiment_config_driven_example.py` | `examples/experiment_config_driven_example.py` | `quantum/quantum1b-research-20260930` |
| `examples/hardware_backend_transpile_noise_example.py` | `examples/hardware_backend_transpile_noise_example.py` | `quantum/quantum1b-research-20260930` |
| `examples/pqk_classification_example.py` | `examples/pqk_classification_example.py` | `quantum/quantum1b-research-20260930` |
| `examples/qae_option_pricing_example.py` | `examples/qae_option_pricing_example.py` | `quantum/quantum1b-research-20260930` |
| `examples/qaoa_portfolio_example.py` | `examples/qaoa_portfolio_example.py` | `quantum/quantum1b-research-20260930` |
| `examples/qrc_time_series_example.py` | `examples/qrc_time_series_example.py` | `quantum/quantum1b-research-20260930` |
| `examples/tail_risk_multi_method_comparison_example.py` | `examples/tail_risk_multi_method_comparison_example.py` | `quantum/quantum1b-research-20260930` |
| `examples/volatility_multi_method_comparison_example.py` | `examples/volatility_multi_method_comparison_example.py` | `quantum/quantum1b-research-20260930` |
| `examples/vqc_qnn_classification_example.py` | `examples/vqc_qnn_classification_example.py` | `quantum/quantum1b-research-20260930` |
| `experiments/__init__.py` | `experiments/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/analysis/__init__.py` | `experiments/analysis/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/configs/__init__.py` | `experiments/configs/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/configs/loader.py` | `experiments/configs/loader.py` | `quantum/quantum1b-research-20260930` |
| `experiments/datasets_interfaces/__init__.py` | `experiments/datasets_interfaces/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/historical/__init__.py` | `experiments/historical/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/parameter_search/__init__.py` | `experiments/parameter_search/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/preprocessing/__init__.py` | `experiments/preprocessing/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/__init__.py` | `experiments/quantum1b/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/calibrate_transpile_optimization_level.py` | `experiments/quantum1b/calibrate_transpile_optimization_level.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/classical_competitor_gate.py` | `experiments/quantum1b/classical_competitor_gate.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/classical_exact.py` | `experiments/quantum1b/classical_exact.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/classical_heuristic.py` | `experiments/quantum1b/classical_heuristic.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/cvar_qaoa.py` | `experiments/quantum1b/cvar_qaoa.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/generators.py` | `experiments/quantum1b/generators.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/hardware_readiness/__init__.py` | `experiments/quantum1b/hardware_readiness/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/hardware_readiness/budgets.py` | `experiments/quantum1b/hardware_readiness/budgets.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/hardware_readiness/dry_run_runbook.py` | `experiments/quantum1b/hardware_readiness/dry_run_runbook.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/hardware_readiness/evidence_store.py` | `experiments/quantum1b/hardware_readiness/evidence_store.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/hardware_readiness/frozen_controls.py` | `experiments/quantum1b/hardware_readiness/frozen_controls.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/hardware_readiness/generate_frozen_controls_snapshot.py` | `experiments/quantum1b/hardware_readiness/generate_frozen_controls_snapshot.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/hardware_readiness/ibm_verifier.py` | `experiments/quantum1b/hardware_readiness/ibm_verifier.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/hardware_readiness/lineage.py` | `experiments/quantum1b/hardware_readiness/lineage.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/hardware_readiness/persistent_budget.py` | `experiments/quantum1b/hardware_readiness/persistent_budget.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/instance.py` | `experiments/quantum1b/instance.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/metrics.py` | `experiments/quantum1b/metrics.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/mixer_warmstart_bench.py` | `experiments/quantum1b/mixer_warmstart_bench.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/noise_validation.py` | `experiments/quantum1b/noise_validation.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/q10_qaoa_reproduction.py` | `experiments/quantum1b/q10_qaoa_reproduction.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/qae_vs_mc.py` | `experiments/quantum1b/qae_vs_mc.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/qrc_esn_harness.py` | `experiments/quantum1b/qrc_esn_harness.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/quantum_lab/__init__.py` | `experiments/quantum1b/quantum_lab/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/quantum_lab/authorization_gate.py` | `experiments/quantum1b/quantum_lab/authorization_gate.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/quantum_lab/candidate_trade_prototype.py` | `experiments/quantum1b/quantum_lab/candidate_trade_prototype.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/quantum_lab/charts.py` | `experiments/quantum1b/quantum_lab/charts.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/quantum_lab/command_center_export.py` | `experiments/quantum1b/quantum_lab/command_center_export.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/quantum_lab/readiness_state.py` | `experiments/quantum1b/quantum_lab/readiness_state.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/quantum_lab/registry.py` | `experiments/quantum1b/quantum_lab/registry.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/qubo_encoding.py` | `experiments/quantum1b/qubo_encoding.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/registry_schema.py` | `experiments/quantum1b/registry_schema.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_a.py` | `experiments/quantum1b/run_track_a.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_b_exp_b1.py` | `experiments/quantum1b/run_track_b_exp_b1.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_b_exp_b2.py` | `experiments/quantum1b/run_track_b_exp_b2.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_b_exp_b3.py` | `experiments/quantum1b/run_track_b_exp_b3.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_c.py` | `experiments/quantum1b/run_track_c.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_d_exp_d1.py` | `experiments/quantum1b/run_track_d_exp_d1.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_d_exp_d1_ladder.py` | `experiments/quantum1b/run_track_d_exp_d1_ladder.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_d_exp_d1_n8.py` | `experiments/quantum1b/run_track_d_exp_d1_n8.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_d_exp_d1_n8_single_shot_probe.py` | `experiments/quantum1b/run_track_d_exp_d1_n8_single_shot_probe.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_d_exp_d1_seed_replication.py` | `experiments/quantum1b/run_track_d_exp_d1_seed_replication.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_e_exp_e1.py` | `experiments/quantum1b/run_track_e_exp_e1.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_e_exp_e2.py` | `experiments/quantum1b/run_track_e_exp_e2.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_e_exp_e3.py` | `experiments/quantum1b/run_track_e_exp_e3.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_e_exp_e4.py` | `experiments/quantum1b/run_track_e_exp_e4.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_e_exp_e4_opt_level_comparison.py` | `experiments/quantum1b/run_track_e_exp_e4_opt_level_comparison.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/run_track_f_exp_f1.py` | `experiments/quantum1b/run_track_f_exp_f1.py` | `quantum/quantum1b-research-20260930` |
| `experiments/quantum1b/synthetic_vol_task.py` | `experiments/quantum1b/synthetic_vol_task.py` | `quantum/quantum1b-research-20260930` |
| `experiments/reporting/__init__.py` | `experiments/reporting/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/reproducibility/__init__.py` | `experiments/reproducibility/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/result_processing/__init__.py` | `experiments/result_processing/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/runners/__init__.py` | `experiments/runners/__init__.py` | `quantum/quantum1b-research-20260930` |
| `experiments/runners/generic_runner.py` | `experiments/runners/generic_runner.py` | `quantum/quantum1b-research-20260930` |
| `experiments/search/__init__.py` | `experiments/search/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/__init__.py` | `hardware/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/backend_discovery/__init__.py` | `hardware/backend_discovery/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/backend_discovery/discovery.py` | `hardware/backend_discovery/discovery.py` | `quantum/quantum1b-research-20260930` |
| `hardware/backend_selection/__init__.py` | `hardware/backend_selection/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/backend_selection/capabilities.py` | `hardware/backend_selection/capabilities.py` | `quantum/quantum1b-research-20260930` |
| `hardware/backend_selection/fallback.py` | `hardware/backend_selection/fallback.py` | `quantum/quantum1b-research-20260930` |
| `hardware/backend_selection/scorer.py` | `hardware/backend_selection/scorer.py` | `quantum/quantum1b-research-20260930` |
| `hardware/backends/__init__.py` | `hardware/backends/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/backends/catalog.py` | `hardware/backends/catalog.py` | `quantum/quantum1b-research-20260930` |
| `hardware/batching/__init__.py` | `hardware/batching/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/batching/batch.py` | `hardware/batching/batch.py` | `quantum/quantum1b-research-20260930` |
| `hardware/braket/__init__.py` | `hardware/braket/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/braket/backend_adapter.py` | `hardware/braket/backend_adapter.py` | `quantum/quantum1b-research-20260930` |
| `hardware/braket/config.py` | `hardware/braket/config.py` | `quantum/quantum1b-research-20260930` |
| `hardware/calibration/__init__.py` | `hardware/calibration/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/calibration/history.py` | `hardware/calibration/history.py` | `quantum/quantum1b-research-20260930` |
| `hardware/calibration/ingest.py` | `hardware/calibration/ingest.py` | `quantum/quantum1b-research-20260930` |
| `hardware/circuit_analysis/__init__.py` | `hardware/circuit_analysis/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/circuit_analysis/compatibility.py` | `hardware/circuit_analysis/compatibility.py` | `quantum/quantum1b-research-20260930` |
| `hardware/circuit_analysis/metrics.py` | `hardware/circuit_analysis/metrics.py` | `quantum/quantum1b-research-20260930` |
| `hardware/coupling_maps/__init__.py` | `hardware/coupling_maps/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/experiment_tracking/__init__.py` | `hardware/experiment_tracking/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/experiment_tracking/tracking.py` | `hardware/experiment_tracking/tracking.py` | `quantum/quantum1b-research-20260930` |
| `hardware/fake_backends/__init__.py` | `hardware/fake_backends/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/ibm_runtime/__init__.py` | `hardware/ibm_runtime/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/ibm_runtime/primitives.py` | `hardware/ibm_runtime/primitives.py` | `quantum/quantum1b-research-20260930` |
| `hardware/jobs/__init__.py` | `hardware/jobs/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/jobs/job.py` | `hardware/jobs/job.py` | `quantum/quantum1b-research-20260930` |
| `hardware/layout/__init__.py` | `hardware/layout/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/mitigation/__init__.py` | `hardware/mitigation/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/mitigation/dd.py` | `hardware/mitigation/dd.py` | `quantum/quantum1b-research-20260930` |
| `hardware/mitigation/pec.py` | `hardware/mitigation/pec.py` | `quantum/quantum1b-research-20260930` |
| `hardware/mitigation/trex.py` | `hardware/mitigation/trex.py` | `quantum/quantum1b-research-20260930` |
| `hardware/noise/__init__.py` | `hardware/noise/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/noise/models.py` | `hardware/noise/models.py` | `quantum/quantum1b-research-20260930` |
| `hardware/noise_learning/__init__.py` | `hardware/noise_learning/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/noise_learning/estimator.py` | `hardware/noise_learning/estimator.py` | `quantum/quantum1b-research-20260930` |
| `hardware/primitives/__init__.py` | `hardware/primitives/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/provenance/__init__.py` | `hardware/provenance/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/provenance/reproducibility_record.py` | `hardware/provenance/reproducibility_record.py` | `quantum/quantum1b-research-20260930` |
| `hardware/provenance/runtime_diagnostics.py` | `hardware/provenance/runtime_diagnostics.py` | `quantum/quantum1b-research-20260930` |
| `hardware/qpu_execution/__init__.py` | `hardware/qpu_execution/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/qpu_execution/execute.py` | `hardware/qpu_execution/execute.py` | `quantum/quantum1b-research-20260930` |
| `hardware/qubit_selection/__init__.py` | `hardware/qubit_selection/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/qubit_selection/selector.py` | `hardware/qubit_selection/selector.py` | `quantum/quantum1b-research-20260930` |
| `hardware/readout_mitigation/__init__.py` | `hardware/readout_mitigation/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/readout_mitigation/m3.py` | `hardware/readout_mitigation/m3.py` | `quantum/quantum1b-research-20260930` |
| `hardware/resource_estimation/__init__.py` | `hardware/resource_estimation/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/resource_estimation/campaign.py` | `hardware/resource_estimation/campaign.py` | `quantum/quantum1b-research-20260930` |
| `hardware/result_processing/__init__.py` | `hardware/result_processing/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/result_processing/braket_normalize.py` | `hardware/result_processing/braket_normalize.py` | `quantum/quantum1b-research-20260930` |
| `hardware/retries/__init__.py` | `hardware/retries/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/retries/retry.py` | `hardware/retries/retry.py` | `quantum/quantum1b-research-20260930` |
| `hardware/routing/__init__.py` | `hardware/routing/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/runtime_versions/__init__.py` | `hardware/runtime_versions/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/runtime_versions/gen2_native/__init__.py` | `hardware/runtime_versions/gen2_native/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/runtime_versions/gen2_native/backend_factory.py` | `hardware/runtime_versions/gen2_native/backend_factory.py` | `quantum/quantum1b-research-20260930` |
| `hardware/runtime_versions/gen2_native/estimator_wrapper.py` | `hardware/runtime_versions/gen2_native/estimator_wrapper.py` | `quantum/quantum1b-research-20260930` |
| `hardware/runtime_versions/gen2_native/noise_learner.py` | `hardware/runtime_versions/gen2_native/noise_learner.py` | `quantum/quantum1b-research-20260930` |
| `hardware/runtime_versions/gen2_native/sampler_wrapper.py` | `hardware/runtime_versions/gen2_native/sampler_wrapper.py` | `quantum/quantum1b-research-20260930` |
| `hardware/sessions/__init__.py` | `hardware/sessions/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/sessions/session.py` | `hardware/sessions/session.py` | `quantum/quantum1b-research-20260930` |
| `hardware/shadow/__init__.py` | `hardware/shadow/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/shadow/shadow_deploy.py` | `hardware/shadow/shadow_deploy.py` | `quantum/quantum1b-research-20260930` |
| `hardware/shadow/shadow_extractor.py` | `hardware/shadow/shadow_extractor.py` | `quantum/quantum1b-research-20260930` |
| `hardware/shadow/shadow_kernel.py` | `hardware/shadow/shadow_kernel.py` | `quantum/quantum1b-research-20260930` |
| `hardware/shadow/tests/__init__.py` | `hardware/shadow/tests/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/shadow/tests/test_shadow.py` | `hardware/shadow/tests/test_shadow.py` | `quantum/quantum1b-research-20260930` |
| `hardware/simulators/__init__.py` | `hardware/simulators/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/simulators/method_selection.py` | `hardware/simulators/method_selection.py` | `quantum/quantum1b-research-20260930` |
| `hardware/tket_transpilation/__init__.py` | `hardware/tket_transpilation/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/topology/__init__.py` | `hardware/topology/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/topology/graph.py` | `hardware/topology/graph.py` | `quantum/quantum1b-research-20260930` |
| `hardware/transpilation/__init__.py` | `hardware/transpilation/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/transpilation/strategies.py` | `hardware/transpilation/strategies.py` | `quantum/quantum1b-research-20260930` |
| `hardware/zne/__init__.py` | `hardware/zne/__init__.py` | `quantum/quantum1b-research-20260930` |
| `hardware/zne/zne.py` | `hardware/zne/zne.py` | `quantum/quantum1b-research-20260930` |
| `integrations/__init__.py` | `integrations/__init__.py` | `quantum/quantum1b-research-20260930` |
| `integrations/heimdall/__init__.py` | `integrations/heimdall/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/__init__.py` | `prototypes/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/experimental/__init__.py` | `prototypes/experimental/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/gen1/__init__.py` | `prototypes/gen1/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/gen2/__init__.py` | `prototypes/gen2/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/hardware/__init__.py` | `prototypes/hardware/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/hardware/generate_qiskit_circuit.py` | `prototypes/hardware/generate_qiskit_circuit.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/hardware/generate_qiskit_circuit_20q.py` | `prototypes/hardware/generate_qiskit_circuit_20q.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/hardware/real_system_validation.py` | `prototypes/hardware/real_system_validation.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/mitigation/__init__.py` | `prototypes/mitigation/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/__init__.py` | `prototypes/other/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/multi_capture_pipeline.py` | `prototypes/other/multi_capture_pipeline.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/phase1_viewer.py` | `prototypes/other/phase1_viewer.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/prepare_dataset_v2.py` | `prototypes/other/prepare_dataset_v2.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/__init__.py` | `prototypes/other/quantum_signal_gen1_infra/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/builder.py` | `prototypes/other/quantum_signal_gen1_infra/builder.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/cli.py` | `prototypes/other/quantum_signal_gen1_infra/cli.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/enrichment.py` | `prototypes/other/quantum_signal_gen1_infra/enrichment.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/ensemble.py` | `prototypes/other/quantum_signal_gen1_infra/ensemble.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/features/__init__.py` | `prototypes/other/quantum_signal_gen1_infra/features/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/features/cross_asset.py` | `prototypes/other/quantum_signal_gen1_infra/features/cross_asset.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/features/macro.py` | `prototypes/other/quantum_signal_gen1_infra/features/macro.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/features/momentum.py` | `prototypes/other/quantum_signal_gen1_infra/features/momentum.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/features/positioning.py` | `prototypes/other/quantum_signal_gen1_infra/features/positioning.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/features/sentiment.py` | `prototypes/other/quantum_signal_gen1_infra/features/sentiment.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/features/volatility.py` | `prototypes/other/quantum_signal_gen1_infra/features/volatility.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/logger.py` | `prototypes/other/quantum_signal_gen1_infra/logger.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/models.py` | `prototypes/other/quantum_signal_gen1_infra/models.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/normalizer.py` | `prototypes/other/quantum_signal_gen1_infra/normalizer.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/reader.py` | `prototypes/other/quantum_signal_gen1_infra/reader.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/quantum_signal_gen1_infra/validator.py` | `prototypes/other/quantum_signal_gen1_infra/validator.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/real_data_pipeline.py` | `prototypes/other/real_data_pipeline.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/redis_capture.py` | `prototypes/other/redis_capture.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/reprocess_features.py` | `prototypes/other/reprocess_features.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/signals_to_dataset.py` | `prototypes/other/signals_to_dataset.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/other/snapshot_to_dataset.py` | `prototypes/other/snapshot_to_dataset.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/phase0_research/__init__.py` | `prototypes/phase0_research/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/phase0_research/config/__init__.py` | `prototypes/phase0_research/config/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/phase0_research/demo/__init__.py` | `prototypes/phase0_research/demo/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/phase0_research/notebooks/__init__.py` | `prototypes/phase0_research/notebooks/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/phase0_research/research_notebooks/__init__.py` | `prototypes/phase0_research/research_notebooks/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/phase1_research/__init__.py` | `prototypes/phase1_research/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/pqk/__init__.py` | `prototypes/pqk/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/pqk/_original_kernel_phase0.py` | `prototypes/pqk/_original_kernel_phase0.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/pqk/p0_1_qrbf_ocsvm_replication/__init__.py` | `prototypes/pqk/p0_1_qrbf_ocsvm_replication/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/pqk/p0_1_qrbf_ocsvm_replication/_data.py` | `prototypes/pqk/p0_1_qrbf_ocsvm_replication/_data.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/pqk/p0_1_qrbf_ocsvm_replication/_kernels.py` | `prototypes/pqk/p0_1_qrbf_ocsvm_replication/_kernels.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/pqk/p0_1_qrbf_ocsvm_replication/run.py` | `prototypes/pqk/p0_1_qrbf_ocsvm_replication/run.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/pqk/phase1b_train.py` | `prototypes/pqk/phase1b_train.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/pqk/phase2b_kernel_train.py` | `prototypes/pqk/phase2b_kernel_train.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qae/__init__.py` | `prototypes/qae/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qae/_compat.py` | `prototypes/qae/_compat.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qae/run_ibm.py` | `prototypes/qae/run_ibm.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qae/run_sim.py` | `prototypes/qae/run_sim.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qae/summarize_results.py` | `prototypes/qae/summarize_results.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qaoa/__init__.py` | `prototypes/qaoa/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qaoa/ablation_fast.py` | `prototypes/qaoa/ablation_fast.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qaoa/n15_overnight.py` | `prototypes/qaoa/n15_overnight.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qaoa/n20_overnight.py` | `prototypes/qaoa/n20_overnight.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qaoa/overnight_sweep.py` | `prototypes/qaoa/overnight_sweep.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qaoa/phase2d_qaoa_optimize.py` | `prototypes/qaoa/phase2d_qaoa_optimize.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/qaoa/qaoa_backtest.py` | `prototypes/qaoa/qaoa_backtest.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/vqc/__init__.py` | `prototypes/vqc/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/vqc/notebooks/__init__.py` | `prototypes/vqc/notebooks/__init__.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/vqc/test_v4_correctness.py` | `prototypes/vqc/test_v4_correctness.py` | `quantum/quantum1b-research-20260930` |
| `prototypes/vqc/vqc_smoke_test.py` | `prototypes/vqc/vqc_smoke_test.py` | `quantum/quantum1b-research-20260930` |
| `tools/__init__.py` | `tools/__init__.py` | `quantum/quantum1b-research-20260930` |
| `tools/benchmark_execution.py` | `tools/benchmark_execution.py` | `quantum/quantum1b-research-20260930` |
| `tools/dataset_preparation.py` | `tools/dataset_preparation.py` | `quantum/quantum1b-research-20260930` |
| `tools/environment_diagnostics.py` | `tools/environment_diagnostics.py` | `quantum/quantum1b-research-20260930` |
| `tools/experiment_launch.py` | `tools/experiment_launch.py` | `quantum/quantum1b-research-20260930` |
| `tools/provenance_tool.py` | `tools/provenance_tool.py` | `quantum/quantum1b-research-20260930` |
| `tools/reporting.py` | `tools/reporting.py` | `quantum/quantum1b-research-20260930` |
| `tools/result_aggregation.py` | `tools/result_aggregation.py` | `quantum/quantum1b-research-20260930` |
| `tools/validation.py` | `tools/validation.py` | `quantum/quantum1b-research-20260930` |
