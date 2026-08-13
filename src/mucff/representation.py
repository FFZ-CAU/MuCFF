"""Aligned and anchor-relative score representations."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def clip_prob(values: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    clean = np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(clean, epsilon, 1.0 - epsilon)


def logit(values: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    probabilities = clip_prob(values, epsilon)
    return np.log(probabilities / (1.0 - probabilities))


def rank_columns(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("Rank input must be a two-dimensional matrix.")
    n_samples = values.shape[0]
    return np.column_stack(
        [rankdata(values[:, index], method="average") / n_samples for index in range(values.shape[1])]
    )


def aligned_state(scores: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    probabilities = clip_prob(scores, epsilon)
    summaries = np.column_stack(
        [
            probabilities.mean(axis=1),
            probabilities.std(axis=1),
            probabilities.min(axis=1),
            probabilities.max(axis=1),
            np.ptp(probabilities, axis=1),
        ]
    )
    return np.hstack(
        [probabilities, rank_columns(probabilities), logit(probabilities, epsilon), summaries]
    ).astype(np.float32)


def select_anchor(scores: np.ndarray, labels: np.ndarray, epsilon: float = 1e-5) -> int:
    probabilities = clip_prob(scores, epsilon)
    aucs = np.asarray(
        [roc_auc_score(labels, probabilities[:, index]) for index in range(probabilities.shape[1])]
    )
    symmetric_auc = np.maximum(aucs, 1.0 - aucs)
    return int(np.argmax(symmetric_auc))


def residual_state(scores: np.ndarray, anchor_index: int, epsilon: float = 1e-5) -> np.ndarray:
    probabilities = clip_prob(scores, epsilon)
    ranks = rank_columns(probabilities)
    logits = logit(probabilities, epsilon)
    anchor_probability = probabilities[:, anchor_index : anchor_index + 1]
    anchor_rank = ranks[:, anchor_index : anchor_index + 1]
    anchor_logit = logits[:, anchor_index : anchor_index + 1]
    delta_probability = probabilities - anchor_probability
    delta_rank = ranks - anchor_rank
    delta_logit = np.clip(logits - anchor_logit, -8.0, 8.0)
    return np.hstack(
        [delta_probability, delta_rank, delta_logit, np.abs(delta_probability)]
    ).astype(np.float32)


def mucff_state(
    scores: np.ndarray,
    anchor_index: int,
    epsilon: float = 1e-5,
) -> np.ndarray:
    return np.hstack(
        [aligned_state(scores, epsilon), residual_state(scores, anchor_index, epsilon)]
    ).astype(np.float32)

