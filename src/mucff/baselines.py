"""Matched score-fusion controls."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .fusion import MuCFFConfig, stable_seed
from .representation import aligned_state, clip_prob, logit, rank_columns


@dataclass(frozen=True)
class BaselinePrediction:
    oof_probability: np.ndarray
    eval_probability: np.ndarray


def reliability_from_auc(aucs: np.ndarray) -> np.ndarray:
    weights = np.maximum(0.0, 2.0 * (np.asarray(aucs, dtype=float) - 0.5))
    if np.all(weights <= 1e-8):
        weights = np.ones_like(weights)
    return weights / weights.sum()


def _linear_pool(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / max(float(weights.sum()), 1e-12)
    return clip_prob(np.dot(scores, normalized))


def _logarithmic_pool(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / max(float(weights.sum()), 1e-12)
    pooled_logit = np.dot(logit(scores), normalized)
    return clip_prob(1.0 / (1.0 + np.exp(-np.clip(pooled_logit, -40.0, 40.0))))


def _dempster_shafer_pool(scores: np.ndarray, reliability: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(reliability, dtype=float), 0.0, 0.98)
    if weights.max() <= 1e-8:
        weights = np.full_like(weights, 1.0 / len(weights))
    weights /= max(float(weights.max()), 1e-12)
    probabilities = clip_prob(scores)
    positive = weights[None, :] * probabilities
    negative = weights[None, :] * (1.0 - probabilities)
    unknown = 1.0 - weights[None, :]
    mass_positive = positive[:, 0].copy()
    mass_negative = negative[:, 0].copy()
    mass_unknown = unknown[:, 0].copy()
    for index in range(1, probabilities.shape[1]):
        conflict = np.clip(
            mass_positive * negative[:, index] + mass_negative * positive[:, index],
            0.0,
            1.0 - 1e-7,
        )
        denominator = 1.0 - conflict
        mass_positive = (
            mass_positive * positive[:, index]
            + mass_positive * unknown[:, index]
            + mass_unknown * positive[:, index]
        ) / denominator
        mass_negative = (
            mass_negative * negative[:, index]
            + mass_negative * unknown[:, index]
            + mass_unknown * negative[:, index]
        ) / denominator
        mass_unknown = mass_unknown * unknown[:, index] / denominator
    return clip_prob(mass_positive + 0.5 * mass_unknown)


def _attention_pool(scores: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    probabilities = clip_prob(scores)
    reliability = weights / max(float(weights.sum()), 1e-12)
    confidence = 2.0 * np.abs(probabilities - 0.5)
    logits = np.log(reliability[None, :] + 1e-12) + alpha * confidence
    logits -= logits.max(axis=1, keepdims=True)
    local_weights = np.exp(logits)
    local_weights /= local_weights.sum(axis=1, keepdims=True)
    return clip_prob((local_weights * probabilities).sum(axis=1))


def _correlation_pruned_indices(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.92,
    maximum: int = 16,
) -> list[int]:
    ranks = rank_columns(scores)
    aucs = np.asarray([roc_auc_score(labels, scores[:, index]) for index in range(scores.shape[1])])
    selected: list[int] = []
    for index in np.argsort(-aucs):
        if not selected:
            selected.append(int(index))
        else:
            correlations = np.asarray(
                [np.corrcoef(ranks[:, index], ranks[:, chosen])[0, 1] for chosen in selected]
            )
            if np.all(np.abs(np.nan_to_num(correlations, nan=1.0)) < threshold):
                selected.append(int(index))
        if len(selected) >= maximum:
            break
    return selected


def _family_best_indices(
    scores: np.ndarray,
    labels: np.ndarray,
    families: tuple[str, ...],
) -> list[int]:
    selected = []
    for family in dict.fromkeys(families):
        indices = [index for index, value in enumerate(families) if value == family]
        aucs = [roc_auc_score(labels, scores[:, index]) for index in indices]
        selected.append(indices[int(np.argmax(aucs))])
    return selected


def _complementarity_matrix(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    correlations = np.corrcoef(scores, rowvar=False)
    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (1.0 - np.abs(correlations)) * np.sqrt(np.outer(weights, weights))
    np.fill_diagonal(matrix, 0.0)
    denominator = float(np.triu(matrix, 1).sum())
    return matrix / denominator if denominator > 1e-12 else matrix


def _choquet_capacity(
    indices: np.ndarray,
    weights: np.ndarray,
    complementarity: np.ndarray,
    gamma: float,
) -> float:
    base = float(weights[indices].sum()) if len(indices) else 0.0
    bonus = (
        float(np.triu(complementarity[np.ix_(indices, indices)], 1).sum())
        if len(indices) >= 2 and gamma > 0
        else 0.0
    )
    return min(1.0, base + gamma * bonus)


def _choquet_integral(
    scores: np.ndarray,
    weights: np.ndarray,
    complementarity: np.ndarray,
    gamma: float,
) -> np.ndarray:
    probabilities = clip_prob(scores)
    output = np.zeros(probabilities.shape[0], dtype=float)
    for row_index, row in enumerate(probabilities):
        order = np.argsort(row)
        ordered = row[order]
        previous = 0.0
        total = 0.0
        for position, value in enumerate(ordered):
            if value > previous:
                total += (value - previous) * _choquet_capacity(
                    order[position:], weights, complementarity, gamma
                )
                previous = value
        output[row_index] = total
    return clip_prob(output)


def fixed_fusion_baselines(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    source_families: tuple[str, ...],
) -> dict[str, BaselinePrediction]:
    oof = clip_prob(oof_scores)
    evaluation = clip_prob(eval_scores)
    aucs = np.asarray([roc_auc_score(labels, oof[:, index]) for index in range(oof.shape[1])])
    reliability = reliability_from_auc(aucs)
    results = {
        "best_source": BaselinePrediction(
            oof[:, int(np.argmax(aucs))], evaluation[:, int(np.argmax(aucs))]
        ),
        "probability_mean": BaselinePrediction(oof.mean(axis=1), evaluation.mean(axis=1)),
        "probability_median": BaselinePrediction(
            np.median(oof, axis=1), np.median(evaluation, axis=1)
        ),
        "rank_mean": BaselinePrediction(
            rank_columns(oof).mean(axis=1), rank_columns(evaluation).mean(axis=1)
        ),
        "majority_vote": BaselinePrediction(
            (oof >= 0.5).mean(axis=1), (evaluation >= 0.5).mean(axis=1)
        ),
        "oof_skill_linear_pool": BaselinePrediction(
            _linear_pool(oof, reliability), _linear_pool(evaluation, reliability)
        ),
        "oof_skill_logarithmic_pool": BaselinePrediction(
            _logarithmic_pool(oof, reliability),
            _logarithmic_pool(evaluation, reliability),
        ),
        "dempster_shafer_pool": BaselinePrediction(
            _dempster_shafer_pool(oof, reliability),
            _dempster_shafer_pool(evaluation, reliability),
        ),
    }

    attention_grid = [0.0, 0.5, 1.0, 2.0, 4.0]
    attention_oof = [_attention_pool(oof, reliability, alpha) for alpha in attention_grid]
    attention_index = int(np.argmax([roc_auc_score(labels, values) for values in attention_oof]))
    results["reliability_attention"] = BaselinePrediction(
        attention_oof[attention_index],
        _attention_pool(evaluation, reliability, attention_grid[attention_index]),
    )

    for label, selected in {
        "family_best_weighted_pool": _family_best_indices(oof, labels, source_families),
        "correlation_pruned_weighted_pool": _correlation_pruned_indices(oof, labels),
    }.items():
        subset_oof = oof[:, selected]
        subset_eval = evaluation[:, selected]
        subset_aucs = np.asarray(
            [roc_auc_score(labels, subset_oof[:, index]) for index in range(subset_oof.shape[1])]
        )
        subset_reliability = reliability_from_auc(subset_aucs)
        results[label] = BaselinePrediction(
            _linear_pool(subset_oof, subset_reliability),
            _linear_pool(subset_eval, subset_reliability),
        )

    indices = _correlation_pruned_indices(oof, labels)[:12]
    selected_oof = oof[:, indices]
    selected_eval = evaluation[:, indices]
    selected_aucs = np.asarray(
        [roc_auc_score(labels, selected_oof[:, index]) for index in range(selected_oof.shape[1])]
    )
    selected_weights = reliability_from_auc(selected_aucs)
    complementarity = _complementarity_matrix(selected_oof, selected_weights)
    gamma_grid = [0.0, 0.05, 0.10, 0.20]
    choquet_oof = [
        _choquet_integral(selected_oof, selected_weights, complementarity, gamma)
        for gamma in gamma_grid
    ]
    gamma_index = int(np.argmax([roc_auc_score(labels, values) for values in choquet_oof]))
    results["choquet_integral"] = BaselinePrediction(
        choquet_oof[gamma_index],
        _choquet_integral(
            selected_eval, selected_weights, complementarity, gamma_grid[gamma_index]
        ),
    )
    return results


def crossfit_logistic_control(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    representation: str,
    config: MuCFFConfig | None = None,
) -> BaselinePrediction:
    settings = config or MuCFFConfig()
    if representation not in {"raw", "aligned"}:
        raise ValueError(f"Unknown representation: {representation}")
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=1800,
            class_weight="balanced",
            random_state=settings.model_seed,
        ),
    )
    splitter = StratifiedKFold(
        settings.outer_folds,
        shuffle=True,
        random_state=stable_seed(settings.seed_base, task_id, "fusion_benchmark_common_cv"),
    )
    oof_probability = np.zeros(labels.size, dtype=np.float32)
    eval_probabilities = []
    for train_index, validation_index in splitter.split(oof_scores, labels):
        if representation == "raw":
            train_state = clip_prob(oof_scores[train_index]).astype(np.float32)
            validation_state = clip_prob(oof_scores[validation_index]).astype(np.float32)
            eval_state = clip_prob(eval_scores).astype(np.float32)
        else:
            train_state = aligned_state(
                oof_scores[train_index], settings.probability_epsilon
            )
            validation_state = aligned_state(
                oof_scores[validation_index], settings.probability_epsilon
            )
            eval_state = aligned_state(eval_scores, settings.probability_epsilon)
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_state, labels[train_index])
        oof_probability[validation_index] = model.predict_proba(validation_state)[:, 1]
        eval_probabilities.append(model.predict_proba(eval_state)[:, 1])
    return BaselinePrediction(
        clip_prob(oof_probability).astype(np.float32),
        clip_prob(np.mean(eval_probabilities, axis=0)).astype(np.float32),
    )
