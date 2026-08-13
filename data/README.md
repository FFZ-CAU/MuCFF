# Evidence ledgers

Each task directory contains one compressed NumPy archive named `evidence_ledger.npz` with the following arrays:

| Array | Shape | Definition |
| --- | --- | --- |
| `task_id` | scalar | Stable task identifier |
| `y_oof` | `(n_train,)` | Labels for the cross-fitted training ledger |
| `y_eval` | `(n_eval,)` | Labels for the fixed evaluation partition |
| `oof_scores` | `(n_train, n_sources)` | Out-of-fold source probabilities |
| `eval_scores` | `(n_eval, n_sources)` | Source probabilities on the fixed evaluation partition |
| `source_ids` | `(n_sources,)` | Source identifiers linked to `source_metadata.csv` |
| `source_families` | `(n_sources,)` | Evidence-family labels |

`dataset_manifest.csv` records task definitions, partition protocols, class counts, and source counts. `ledger_manifest.csv` provides the machine-readable ledger index.

`dataset_sources.csv` lists the upstream benchmark resources. `model_manifest.csv` lists the foundation models and DNA-shape software represented in the ledgers. The repository contains labels and derived source probabilities; it does not redistribute pretrained weights or raw benchmark sequences.

All source probabilities are stored in the same column order for the OOF and evaluation matrices within a task.
