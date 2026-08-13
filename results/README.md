# Result index

| Manuscript evidence | Public command or artifact |
| --- | --- |
| Ten-task MuCFF metrics | `mucff run --data data/processed --config configs/main_experiment.json --output outputs/main_experiment` |
| Prediction-level numerical verification | `mucff verify --output outputs/main_experiment --reference results/reference/reference_predictions.npz` |
| Framework-level comparison | `reference/framework_comparison.csv` |
| Fusion-operator comparison | `reference/fusion_operator_comparison.csv` |
| Score-representation analysis | `reference/score_representation_analysis.csv` |
| Foundation-evidence ablation | `reference/foundation_evidence_ablation.csv` |
| Cumulative evidence construction | `reference/cumulative_evidence_construction.csv` |
| Fusion-head sensitivity | `reference/fusion_head_sensitivity.csv` |
| Taskwise mechanism decomposition | `reference/taskwise_mechanism_decomposition.csv` |
| Source inventory and performance | `reference/source_performance_summary.csv` |
| MuCFF versus sparse aligned control | `reference/reference_metric_summary.csv` |

The `reference_predictions.npz` archive contains the task labels and the OOF and evaluation probabilities for MuCFF and the sparse aligned control.
