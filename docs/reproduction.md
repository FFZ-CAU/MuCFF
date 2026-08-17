# Reproduction

## Environment

The reference analysis uses Python 3.11 on Linux with the versions in `requirements-lock.txt`. MuCFF is CPU compatible. `xgb_threads` is set to 2 in the released configuration; numerical-library thread limits can be set with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`.

## Primary analysis

```bash
mucff run --data data/ledgers --config configs/main_experiment.json --output outputs/main_experiment
mucff verify --output outputs/main_experiment --reference results/reference/reference_predictions.npz
```

The default command evaluates 19 run ledgers and aggregates them into ten equally weighted tasks: five GUE tasks, four iPromoter tasks, and one Rice 6mA task pooled across ten outer test folds. Expected primary means are AUC 0.929801, AUPRC 0.935494, MCC 0.666422, and accuracy 0.825518. The aligned logistic control has mean AUC 0.924627.

Run or verify one task family with `--tasks`. For Rice 6mA, the family identifier expands to all ten fold ledgers:

```bash
mucff run --data data/ledgers --config configs/main_experiment.json --output outputs/rice_6ma --tasks snnrice6ma_rice_chen
mucff verify --output outputs/rice_6ma --reference results/reference/reference_predictions.npz --tasks snnrice6ma_rice_chen
```

## Auxiliary enhancer tasks

```bash
mucff run --data data/ledgers --config configs/main_experiment.json --output outputs/ienhancer --tasks ienhancer_recognition ienhancer_strength
```

These tasks use the original uncorrected benchmark partitions and do not enter the ten-task aggregate.

## Comparators and robustness

```bash
python -m pip install -r requirements-extended-lock.txt
python -m pip install -e ".[benchmark,figures]"
mucff benchmark --data data/ledgers --config configs/main_experiment.json --output outputs/extended_baselines --device cpu
mucff robustness --data data/ledgers --config configs/main_experiment.json --output outputs/robustness
python scripts/run_evidence_growth.py --data data/ledgers --output outputs/evidence_growth
mucff plot --reference results/reference --output outputs/figures
```

Install the attention environment and run the matched QKV controls separately:

```bash
python -m pip install -r requirements-attention-lock.txt
mucff benchmark --suite attention --data data/ledgers --config configs/main_experiment.json --output outputs/qkv --device cpu
```

Same-ledger fusion controls use identical OOF and evaluation score matrices. Task-model literature comparisons are indexed separately in `results/literature_context/comparison_protocol_registry.csv`, together with their data partition and selection protocol.

Reference prediction arrays were generated with the locked Linux environment. Verification requires identical labels, a maximum probability difference no larger than `0.03`, a mean probability difference no larger than `0.003`, and an AUC difference no larger than `5e-4` for every method and partition.
