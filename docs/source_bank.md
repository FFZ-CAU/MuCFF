# Evidence source bank

The source inventory uses stable identifiers from `data/source_metadata.csv`.

| Source identifiers | Evidence | Implementation or resource |
| --- | --- | --- |
| P01, P02, P04, P20 | DNA-shape profiles | DNAshapeR resource in `data/model_manifest.csv`; cross-fitted heads in `mucff.source_models` |
| P03, P07, P09, P10, P17, P21 | Genomic foundation-model embeddings | Model identifiers in `data/model_manifest.csv`; cross-fitted heads in `mucff.source_models` |
| P05, P06, P08, P11--P19, P22--P30 | Composition, FCGR, motif, position, CKSNAP, Z-curve, EIIP, and dinucleotide descriptors | `mucff.features` and `mucff.source_models` |
| E01--E05 | Reverse-complement motif and biophysical grammar experts | `mucff.neural_sources.ReverseComplementGrammarExpert` |
| D03--D20 | Pre-classifier local fusion, cross-foundation interaction, and pooling-interaction channels | `mucff.neural_sources.CrossSourceEmbeddingFusion` and the source definitions in `data/source_metadata.csv` |
| D01--D02 | Cross-fitted meta-evidence and score-residual evidence | OOF source interface in `mucff.ledger`, aligned representations in `mucff.representation`, and integration models in `mucff.fusion` |

Each reported fusion task starts from sample-aligned OOF and evaluation probabilities in `data/processed`. The archives contain labels, scores, source identifiers, and source-family labels. Foundation-model weights and raw benchmark sequences are obtained from the listed upstream resources; DNA-shape tracks are generated with the DNAshapeR resource identified in `data/model_manifest.csv`.
