# Method specification

For a task with `M` evidence sources, MuCFF receives an OOF score matrix and a fixed-partition evaluation score matrix.

## Aligned evidence

Each source probability is represented by its clipped probability, percentile rank, and logit. Five row-level summaries are appended: mean, standard deviation, minimum, maximum, and range. The aligned state has `3M + 5` dimensions.

## Anchor-relative evidence

Within each outer training fold, the source with the largest orientation-insensitive OOF AUC is selected as the anchor. Source skill, rank-based nonredundancy, and rescue-versus-harm behavior define a fold-local routing prior. For each sample, family agreement, source confidence, anchor uncertainty, and source-anchor conflict modulate this prior. The resulting 12-dimensional routed state summarizes signed probability, rank, and logit innovations, disagreement magnitude, directional support, and uncertainty interactions.

## Joint sparse decision

The aligned and routed states are concatenated into a `3M + 17` dimensional vector. Routing-state estimation, standardization, and an elastic-net logistic model are fitted only on the outer-fold training subset. Held-out OOF probabilities determine the classification threshold; evaluation labels are used only for metric calculation.

The reported configuration uses four outer folds, `C = 0.03`, an L1 ratio of `0.5`, class balancing, and 2,200 SAGA iterations.
