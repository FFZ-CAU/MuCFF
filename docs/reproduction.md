# Reproduction

## Environment

The reference analysis uses Python 3.11 and the versions in `requirements-lock.txt`. The complete ten-task fusion run is CPU compatible. Setting `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 2 limits numerical-library parallelism.

## Quick verification

```bash
mucff run --data data/processed --config configs/main_experiment.json --output outputs/rice_6ma --tasks snnrice6ma_rice_chen
mucff verify --output outputs/rice_6ma --reference results/reference/reference_predictions.npz --tasks snnrice6ma_rice_chen
```

## Complete analysis

```bash
mucff run --data data/processed --config configs/main_experiment.json --output outputs/main_experiment
mucff verify --output outputs/main_experiment --reference results/reference/reference_predictions.npz
python scripts/export_reference_results.py --run outputs/main_experiment --output outputs/reproduced_reference
```

The complete command processes ten tasks and writes task-level metrics, OOF and evaluation predictions, across-task summaries, and paired AUC comparisons. A reference CPU run with two numerical-library threads required approximately 13 minutes and less than 1 GB of working memory; installation and the 52 MB processed ledger collection require additional disk space.

The expected MuCFF means are AUC 0.943950, AUPRC 0.945048, MCC 0.750372, and accuracy 0.873485. The matched sparse aligned-state control has mean AUC 0.943705. Prediction verification uses an absolute tolerance of `2e-7`.

The auxiliary enhancer-recognition and enhancer-strength tasks use a separate
30-source ledger and do not enter the primary ten-task aggregate:

```bash
mucff run --data data/auxiliary --config configs/main_experiment.json --output outputs/auxiliary_enhancer
```

## Extended analyses

```bash
python -m pip install -r requirements-extended-lock.txt
python -m pip install -e ".[benchmark,figures,sources]"
mucff benchmark --data data/processed --config configs/main_experiment.json --output outputs/extended_baselines --device cpu
mucff plot --reference results/reference --output outputs/figures
mucff robustness --data data/processed --config configs/main_experiment.json --output outputs/robustness
```

The robustness analysis uses the same four fusion folds and fixed MuCFF configuration. Source-group ablations refit the fusion model after removing a complete source group. Deployment-shift analyses retain clean training and replace unavailable evaluation scores by fold-training medians, reverse the task-level OOF-selected strongest source for a conflict test, or add deterministic Gaussian score noise. Missing-source rank coordinates use the empirical midpoint.

The benchmark command adds XGBoost, LightGBM, histogram gradient boosting, multilayer perceptron, kernel approximation, Super Learner, and matched logistic controls. The plotting command writes PDF, SVG, and PNG figures from the released result tables.
