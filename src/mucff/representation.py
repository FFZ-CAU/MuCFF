"""Aligned and anchor-relative score representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class RoutingState:
    anchor_index: int
    prior: np.ndarray
    family_indices: tuple[np.ndarray, ...]


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


def fit_routing_state(
    scores: np.ndarray,
    labels: np.ndarray,
    source_families: tuple[str, ...] | list[str],
    epsilon: float = 1e-5,
) -> RoutingState:
    probabilities = clip_prob(scores, epsilon)
    ranks = rank_columns(probabilities)
    aucs = np.asarray(
        [roc_auc_score(labels, probabilities[:, index]) for index in range(probabilities.shape[1])]
    )
    skill = np.clip(2.0 * np.maximum(aucs, 1.0 - aucs) - 1.0, 0.01, 1.0)
    anchor_index = int(np.argmax(np.maximum(aucs, 1.0 - aucs)))

    anchor_correct = (probabilities[:, anchor_index] >= 0.5) == labels
    source_correct = (probabilities >= 0.5) == labels[:, None]
    rescue = np.mean((~anchor_correct[:, None]) & source_correct, axis=0)
    harm = np.mean(anchor_correct[:, None] & (~source_correct), axis=0)
    rescue_ratio = (rescue + 0.005) / (rescue + harm + 0.01)

    correlation = np.corrcoef(ranks, rowvar=False)
    correlation = np.atleast_2d(np.nan_to_num(np.abs(correlation), nan=1.0))
    np.fill_diagonal(correlation, 0.0)
    uniqueness = np.clip(1.0 - correlation.mean(axis=1), 0.02, 1.0)
    prior = skill * np.sqrt(uniqueness) * (0.35 + 0.65 * rescue_ratio)
    prior[anchor_index] = 0.0
    if prior.sum() <= epsilon:
        prior[:] = 1.0
        prior[anchor_index] = 0.0
    prior /= prior.sum()

    if len(source_families) != probabilities.shape[1]:
        raise ValueError("Source-family labels must match the score columns.")
    family_order = tuple(dict.fromkeys(source_families))
    family_indices = tuple(
        np.asarray(
            [index for index, family in enumerate(source_families) if family == name],
            dtype=int,
        )
        for name in family_order
    )
    return RoutingState(anchor_index, prior.astype(np.float64), family_indices)


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


def compact_residual_state(
    scores: np.ndarray,
    routing: RoutingState,
    temperature: float = 0.20,
    epsilon: float = 1e-5,
) -> np.ndarray:
    probabilities = clip_prob(scores, epsilon)
    ranks = rank_columns(probabilities)
    logits = logit(probabilities, epsilon)
    anchor = probabilities[:, routing.anchor_index : routing.anchor_index + 1]
    anchor_rank = ranks[:, routing.anchor_index : routing.anchor_index + 1]
    anchor_logit = logits[:, routing.anchor_index : routing.anchor_index + 1]
    delta_probability = probabilities - anchor
    delta_rank = ranks - anchor_rank
    delta_logit = np.clip(logits - anchor_logit, -8.0, 8.0)
    absolute_delta = np.abs(delta_probability)

    family_centres = np.zeros_like(probabilities)
    for indices in routing.family_indices:
        family_centres[:, indices] = probabilities[:, indices].mean(axis=1, keepdims=True)
    family_support = np.exp(
        -np.abs(probabilities - family_centres) / max(temperature, epsilon)
    )
    source_confidence = 2.0 * np.abs(probabilities - 0.5)
    anchor_uncertainty = 4.0 * anchor * (1.0 - anchor)
    disagreement = probabilities.std(axis=1, keepdims=True)
    conflict = ((probabilities >= 0.5) != (anchor >= 0.5)).astype(float)

    local_gate = routing.prior[None, :] * (0.20 + 0.80 * family_support)
    local_gate *= 0.20 + 0.80 * source_confidence
    local_gate *= 0.30 + 0.70 * anchor_uncertainty
    local_gate[:, routing.anchor_index] = 0.0
    local_gate /= np.maximum(local_gate.sum(axis=1, keepdims=True), epsilon)
    conflict_gate = local_gate * (0.25 + 0.75 * conflict)
    conflict_gate /= np.maximum(conflict_gate.sum(axis=1, keepdims=True), epsilon)

    routed_probability = (local_gate * delta_probability).sum(axis=1)
    conflict_probability = (conflict_gate * delta_probability).sum(axis=1)
    positive = np.maximum(delta_probability, 0.0)
    negative = np.minimum(delta_probability, 0.0)
    return np.column_stack(
        [
            routed_probability,
            (local_gate * delta_rank).sum(axis=1),
            (local_gate * delta_logit).sum(axis=1),
            (local_gate * absolute_delta).sum(axis=1),
            conflict_probability,
            (conflict_gate * delta_logit).sum(axis=1),
            (local_gate * positive).sum(axis=1),
            (local_gate * negative).sum(axis=1),
            anchor_uncertainty[:, 0],
            disagreement[:, 0],
            routed_probability * anchor_uncertainty[:, 0],
            conflict_probability * disagreement[:, 0],
        ]
    ).astype(np.float32)


def mucff_state(
    scores: np.ndarray,
    routing: RoutingState,
    temperature: float = 0.20,
    epsilon: float = 1e-5,
) -> np.ndarray:
    return np.hstack(
        [
            aligned_state(scores, epsilon),
            compact_residual_state(scores, routing, temperature, epsilon),
        ]
    ).astype(np.float32)
