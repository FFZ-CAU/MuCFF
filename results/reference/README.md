# Reference results

This directory contains the numerical evidence used in the manuscript. The
root-level tables and prediction archives define the primary ten-task result;
the subdirectories separate supporting analyses by scientific role.

## Primary analysis

- `reference_predictions.npz`: labels and OOF/evaluation predictions for the
  released main run.
- `reference_task_metrics.csv`, `reference_method_summary.csv`, and
  `reference_metric_summary.csv`: direct outputs and the matched comparison
  with the sparse aligned control.
- `complete_method_task_metrics.csv`: MuCFF metrics for each primary task.
- `framework_task_metrics.csv` and `framework_method_summary.csv`: the full
  common-ledger comparison.
- `matched_auc_comparisons.csv`: paired task-level AUC comparisons, bootstrap
  intervals, wins/ties/losses, and Wilcoxon probabilities.
- `evidence_growth_*.csv`: ordered construction of the evidence bank.
- `residual_state_ablation_*.csv`: aligned, compact-residual, and complete
  source-preserving residual states under matched decision settings.

## Supporting analyses

- `mechanism_ablation/`: the matched 2 x 2 residual-state and regularization
  experiment.
- `representation_analysis/`: state separability and coefficient-contribution
  summaries used in the representation figure.
- `robustness/`: source-group refitting and fixed-model deployment
  perturbations.
- `source_inventory/`: task-level performance of every evidence source.
- `task_model_reruns/`: architecture-matched iPromoter-SeqVec and SNNRice6mA
  reruns on the fixed manifests.
- `literature_context/`: protocol registry and published task-model values.
- `auxiliary/`: enhancer-recognition and enhancer-strength results, which do
  not enter the primary ten-task aggregate.

## QKV comparison

`reference_attention_predictions.npz` and the corresponding attention tables
contain the source self-attention and anchor-query cross-attention controls,
together with MuCFF and the sparse aligned control on the same evidence ledger.
