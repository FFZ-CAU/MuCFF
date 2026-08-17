# Result index

## Same-ledger experiments

| Evidence | Artifact |
| --- | --- |
| Final task metrics and primary aggregate | `reference/selected_framework_metrics.csv`, `reference/selected_framework_summary.csv` |
| Released prediction arrays | `reference/reference_predictions.npz` |
| Source-level performance | `reference/single_source_metrics_by_run.csv`, `reference/single_source_summary.csv` |
| OOF-selected source controls | `reference/oof_selected_single_source_by_task.csv`, `reference/oof_selected_standalone_source_by_task.csv` |
| Fusion comparator summary | `reference/matched_framework_comparisons.csv` |
| Fusion-operator metrics | `reference/fusion_operator_task_metrics.csv`, `reference/fusion_operator_summary.csv` |
| Evidence-bank growth | `reference/evidence_growth_run_metrics.csv`, `reference/evidence_growth_task_metrics.csv`, `reference/evidence_growth_summary.csv` |
| Dual-decision weight sensitivity | `reference/dual_decision_results.csv` |
| Source refitting ablations | `reference/source_refit_task_metrics.csv`, `reference/source_refit_summary.csv` |
| Missing and perturbed source tests | `reference/deployment_robustness_task_metrics.csv`, `reference/deployment_robustness_summary.csv` |

The reference archive stores labels and OOF/evaluation probabilities for `aligned_logistic_l2`, `routed_logistic_l2`, `aligned_xgboost`, and `mucff` for every run ledger.

## Task-model context

`literature_context/comparison_protocol_registry.csv` records the dataset partition, model-selection unit, and comparison status for published task models. Published values and protocol-matched reruns remain separate from same-ledger fusion controls because their training and selection procedures are not interchangeable.
