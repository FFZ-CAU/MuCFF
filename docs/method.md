# Method specification

For a task with `M` evidence sources, MuCFF receives an OOF training-score matrix and a fixed-partition evaluation-score matrix. Every fitting operation used to construct the fusion state is repeated inside the fusion cross-validation folds.

## Aligned score state

Each source is represented by its clipped probability, empirical percentile rank, and clipped logit. Five sample-level summaries are appended: mean, standard deviation, minimum, maximum, and range. The aligned state therefore has `3M + 5` coordinates.

## Complementary routing state

Within each fusion fold, the source with the highest orientation-insensitive training AUC is the anchor. Source skill, rank nonredundancy, and rescue-versus-harm behavior define fold-specific routing priors. Samplewise confidence, anchor uncertainty, family agreement, and source-anchor conflict then modulate signed probability, rank, and logit innovations. Twelve aggregated coordinates retain positive and negative support, disagreement magnitude, and uncertainty interactions without concatenating every pairwise residual.

## Dual decision integration

The routed path fits a standardized L2 logistic model to the concatenated aligned and complementary-routing state. In parallel, an XGBoost path models nonlinear interactions in the aligned state. MuCFF combines their probabilities as

```text
sigmoid(0.70 * logit(p_routed) + 0.30 * logit(p_nonlinear)).
```

The fixed configuration uses four fusion folds, logistic `C = 0.10`, routing temperature `0.20`, and 260 depth-3 boosting trees. Classification thresholds maximize OOF MCC over the configured quantile grid and are then applied to the evaluation partition.

## Evaluation unit

Five GUE tasks and four iPromoter tasks use their released fixed evaluation partitions. Rice 6mA uses ten outer test folds; fold predictions are pooled before computing one Rice task result. These ten task-level results receive equal weight in cross-task summaries. Enhancer recognition and enhancer strength are evaluated separately on their original uncorrected partitions.

The MuCFF configuration is selected by mean OOF AUC across the ten primary tasks. Evaluation labels are used only for the final reported metrics.
