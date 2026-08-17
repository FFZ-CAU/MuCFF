# MuCFF

MuCFF implements Multisource Complementary Feature Fusion for functional DNA sequence classification. The framework converts heterogeneous source predictions into an out-of-fold evidence ledger, aligns their probability, rank, and logit coordinates, models anchor-relative complementary evidence, and combines routed linear and nonlinear decisions in logit space.

Repository: https://github.com/FFZ-CAU/MuCFF

## Installation

Python 3.10 or later is required. The reference environment is specified in `requirements-lock.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Reproduction

Run the ten primary evaluation tasks from the repository root:

```bash
mucff run --data data/ledgers --config configs/main_experiment.json --output outputs/main_experiment
mucff verify --output outputs/main_experiment --reference results/reference/reference_predictions.npz
```

Rice 6mA uses ten outer test folds. Its task name expands to all ten runs and is pooled once in the task-level summary:

```bash
mucff run --data data/ledgers --config configs/main_experiment.json --output outputs/rice_6ma --tasks snnrice6ma_rice_chen
mucff verify --output outputs/rice_6ma --reference results/reference/reference_predictions.npz --tasks snnrice6ma_rice_chen
```

The enhancer-recognition and enhancer-strength evaluations use the original uncorrected benchmark partitions and are reported as auxiliary tasks:

```bash
mucff run --data data/ledgers --config configs/main_experiment.json --output outputs/ienhancer --tasks ienhancer_recognition ienhancer_strength
```

The primary aggregate has mean AUC 0.929801 and mean AUPRC 0.935494. The matched aligned logistic control obtains 0.924627 AUC and 0.928843 AUPRC. Commands write run-level predictions, ten-task metrics, method summaries, and paired AUC comparisons. Use `--resume` to retain completed run directories.

## Additional analyses

```bash
python -m pip install -r requirements-extended-lock.txt
python -m pip install -e ".[benchmark,figures]"
mucff benchmark --data data/ledgers --config configs/main_experiment.json --output outputs/extended_baselines --device cpu
mucff robustness --data data/ledgers --config configs/main_experiment.json --output outputs/robustness
mucff plot --reference results/reference --output outputs/figures
```

The matched QKV comparison is available after installing `requirements-attention-lock.txt` and running `mucff benchmark --suite attention`. Literature values and protocol-matched architecture reruns are kept in `results/literature_context`; they are separate from same-ledger fusion comparisons.

## Repository contents

- `configs`: MuCFF and evaluation settings.
- `data/ledgers`: OOF training scores and fixed-partition evaluation scores.
- `data/source_metadata.csv`: evidence-source definitions and families.
- `data/dataset_manifest.csv`: task roles, partitions, and sample counts.
- `src/mucff`: representations, fusion models, controls, metrics, and statistics.
- `results/reference`: manuscript result tables and reference predictions.
- `results/literature_context`: published results, protocol registry, and matched reruns.
- `tests`: ledger, representation, robustness, and numerical regression tests.

Details of the method, evidence bank, and execution contract are provided in `docs/method.md`, `docs/source_bank.md`, and `docs/reproduction.md`.

## Tests

```bash
python -m pytest
```

## Citation

Wentao Gong, Jiaying Zhang, Jingrui Li, Huihui Pang, and Feifan Zhang. A multisource complementary feature fusion method for improving functional DNA sequence classification.

## License

The MuCFF source code is released under the MIT License. Benchmark data, derived evidence ledgers, pretrained models, and model-derived artifacts retain the terms of their upstream resources.
