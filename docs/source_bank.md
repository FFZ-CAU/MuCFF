# Evidence source bank

The complete source inventory is defined in `data/source_metadata.csv`. Most ledgers contain 44 sources; the GUE mouse task contains 40 because four unavailable channels are omitted. The metadata table contains 46 unique definitions because two promoter-specific motif heads replace their generic counterparts on the applicable tasks.

| Source tier | Sources per complete ledger | Evidence represented |
| --- | ---: | --- |
| Primitive | 28 | k-mer composition, FCGR, motif and position, CKSNAP, Z-curve, EIIP and dinucleotide autocorrelation, centered context, DNABERT-1, DNABERT-2, Nucleotide Transformer, and DNAshapeR |
| Foundation interaction | 5 | within-model pooling, cross-foundation attention, and anchor-relative foundation evidence |
| Local or derived | 6 | aligned-score and local cross-foundation prediction channels |
| Sequence grammar | 5 | reverse-complement multiscale, biophysical, and position-aware grammar experts |

The evidence ledger is a sample-aligned interface. Each source head is fitted on training data and contributes OOF probabilities for fusion fitting plus probabilities for the fixed evaluation partition. The fusion layer receives no raw sequence or embedding tensor.

Foundation-model identifiers and DNAshapeR resources are listed in `data/model_manifest.csv`. Upstream benchmark locations and citations are listed in `data/dataset_sources.csv`. The repository contains the derived score ledgers used for the reported fusion experiments.
