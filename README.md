# MuCFF

MuCFF is the reference implementation of Multisource Complementary Feature Fusion for functional DNA sequence classification. It integrates cross-fitted evidence through score alignment, reliability-guided residual routing, and a joint sparse decision model.

Repository: https://github.com/FFZ-CAU/MuCFF

## Installation

Python 3.10 or later is required. The reported fusion experiments use the dependency versions listed in `requirements-lock.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Reproduction

The complete ten-task experiment is run from the repository root:

```bash
mucff run --data data/processed --config configs/main_experiment.json --output outputs/main_experiment
```

An interrupted complete run can be continued with `--resume`.

A single task can be run with:

```bash
mucff run --data data/processed --config configs/main_experiment.json --output outputs/rice_6ma --tasks snnrice6ma_rice_chen
mucff verify --output outputs/rice_6ma --reference results/reference/reference_predictions.npz --tasks snnrice6ma_rice_chen
```

The two auxiliary iEnhancer tasks use a separate 30-source evidence specification
and are excluded from the primary aggregate. They are reproduced independently:

```bash
mucff run --data data/auxiliary --config configs/main_experiment.json --output outputs/auxiliary_enhancer
```

The command writes task-level predictions and metrics, an across-task method summary, and matched AUC comparisons. Reference tables used in the manuscript are provided in `results/reference`.

Verify a complete run against the released reference predictions:

```bash
mucff verify --output outputs/main_experiment --reference results/reference/reference_predictions.npz
```

Extended learned-fusion baselines and quantitative figures are generated with:

```bash
python -m pip install -r requirements-extended-lock.txt
python -m pip install -e ".[benchmark,figures,sources]"
mucff benchmark --data data/processed --config configs/main_experiment.json --output outputs/extended_baselines --device cuda
mucff plot --reference results/reference --output outputs/figures
mucff robustness --data data/processed --config configs/main_experiment.json --output outputs/robustness
```

The robustness command reports source-group removal with refitting and fixed-model deployment tests for missing, conflicting, and noisy source scores.

## Repository contents

- `configs`: reported fusion and evaluation settings.
- `data/processed`: cross-fitted training scores and fixed-partition evaluation scores.
- `data/source_metadata.csv`: stable source identifiers, evidence types, and prediction heads.
- `data/auxiliary_source_metadata.csv`: stable source identifiers for the auxiliary enhancer tasks.
- `data/dataset_sources.csv` and `data/model_manifest.csv`: upstream data and representation resources.
- `src/mucff`: evidence representations, MuCFF, matched controls, source modules, metrics, and statistics.
- `tests`: ledger, representation, and numerical regression tests.
- `results/reference`: manuscript result tables and reference predictions.

The released evidence ledgers are the reproducibility boundary for all reported fusion, comparator, ablation, and robustness analyses. Their arrays and dimensions are documented in `data/README.md`. Source definitions and extraction modules document upstream evidence construction; raw sequences and pretrained weights are obtained from the cited upstream resources.

The complete execution contract is provided in `docs/reproduction.md`. Evidence-source families and implementation modules are indexed in `docs/source_bank.md`.

## Tests

```bash
python -m pytest
```

## Citation

Please cite:

Wentao Gong, Jiaying Zhang, Jingrui Li, Huihui Pang, and Feifan Zhang. A multisource complementary feature fusion method for improving functional DNA sequence classification.

## License

The MuCFF source code is released under the MIT License. Benchmark data,
derived evidence ledgers, pretrained models, and model-derived artifacts retain
the terms of their upstream resources and are not relicensed by this software
license.
