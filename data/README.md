# Evidence ledgers

Each directory under `ledgers/` contains `expanded_ledger.npz` with the following arrays:

| Array | Shape | Definition |
| --- | --- | --- |
| `y_oof` | `(n_train,)` | Labels aligned to the cross-fitted training scores |
| `y_test` | `(n_eval,)` | Labels for the fixed evaluation partition |
| `oof_scores` | `(n_train, n_sources)` | OOF source probabilities |
| `test_scores` | `(n_eval, n_sources)` | Source probabilities on the evaluation partition |
| `source_names` | `(n_sources,)` | Stable identifiers from `source_metadata.csv` |
| `source_families` | `(n_sources,)` | Evidence-family labels |

`ledger_manifest.csv` indexes all 21 run ledgers. Nineteen primary runs represent ten evaluation tasks because Rice 6mA has ten outer test folds. The enhancer-recognition and enhancer-strength ledgers have the `Auxiliary` role and are indexed separately in `auxiliary_ledger_manifest.csv`.

`dataset_manifest.csv` records task definitions, released partition protocols, class counts, and source counts. `dataset_sources.csv` and `model_manifest.csv` identify the upstream benchmark and representation resources.

OOF and evaluation matrices use the same source-column order within each task. The loader accepts the public array names above and exposes them through the unified `EvidenceLedger` interface.

The MIT License applies to the software. Benchmark data, derived evidence ledgers, pretrained models, and model-derived artifacts retain the terms of their upstream resources.
