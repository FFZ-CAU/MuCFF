"""Cross-fitted MuCFF estimator."""

from __future__ import annotations

import warnings
import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .representation import aligned_state, clip_prob, fit_routing_state, mucff_state


@dataclass(frozen=True)
class MuCFFConfig:
    outer_folds: int = 4
    seed_base: int = 20260806
    model_seed: int = 20260807
    regularization_c: float = 0.03
    l1_ratio: float = 0.5
    max_iterations: int = 2200
    probability_epsilon: float = 1e-5
    routing_temperature: float = 0.20


@dataclass(frozen=True)
class FusionPredictions:
    oof_probability: np.ndarray
    eval_probability: np.ndarray
    anchor_indices: tuple[int, ...]
    residual_nonzero_fraction: float


def stable_seed(base: int, *parts: object) -> int:
    token = "|".join(map(str, parts)).encode("utf-8")
    return base + int(hashlib.sha256(token).hexdigest()[:8], 16) % 1_000_000


def make_sparse_decision(config: MuCFFConfig):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.regularization_c,
            solver="saga",
            l1_ratio=config.l1_ratio,
            max_iter=config.max_iterations,
            class_weight="balanced",
            random_state=config.model_seed,
        ),
    )


def predict_probability(model, features: np.ndarray, epsilon: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = model.predict_proba(features)[:, 1]
    return clip_prob(values, epsilon)


def _crossfit(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    config: MuCFFConfig,
    include_residual: bool,
    source_families: tuple[str, ...] = (),
) -> FusionPredictions:
    splitter = StratifiedKFold(
        config.outer_folds,
        shuffle=True,
        random_state=stable_seed(config.seed_base, task_id, "fusion_benchmark_common_cv"),
    )
    oof_probability = np.zeros(labels.size, dtype=np.float32)
    eval_probabilities: list[np.ndarray] = []
    anchors: list[int] = []
    nonzero: list[float] = []
    aligned_dimension = 3 * oof_scores.shape[1] + 5

    for train_index, validation_index in splitter.split(oof_scores, labels):
        if include_residual:
            routing = fit_routing_state(
                oof_scores[train_index],
                labels[train_index],
                source_families,
                config.probability_epsilon,
            )
            train_state = mucff_state(
                oof_scores[train_index],
                routing,
                config.routing_temperature,
                config.probability_epsilon,
            )
            validation_state = mucff_state(
                oof_scores[validation_index],
                routing,
                config.routing_temperature,
                config.probability_epsilon,
            )
            eval_state = mucff_state(
                eval_scores,
                routing,
                config.routing_temperature,
                config.probability_epsilon,
            )
        else:
            train_state = aligned_state(oof_scores[train_index], config.probability_epsilon)
            validation_state = aligned_state(oof_scores[validation_index], config.probability_epsilon)
            eval_state = aligned_state(eval_scores, config.probability_epsilon)

        model = clone(make_sparse_decision(config))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_state, labels[train_index])
        oof_probability[validation_index] = predict_probability(
            model, validation_state, config.probability_epsilon
        )
        eval_probabilities.append(predict_probability(model, eval_state, config.probability_epsilon))
        if include_residual:
            anchors.append(routing.anchor_index)
        if include_residual:
            coefficients = model.named_steps["logisticregression"].coef_[0]
            nonzero.append(float(np.mean(np.abs(coefficients[aligned_dimension:]) > 1e-9)))

    return FusionPredictions(
        oof_probability=clip_prob(oof_probability, config.probability_epsilon).astype(np.float32),
        eval_probability=clip_prob(
            np.mean(eval_probabilities, axis=0), config.probability_epsilon
        ).astype(np.float32),
        anchor_indices=tuple(anchors),
        residual_nonzero_fraction=float(np.mean(nonzero)) if nonzero else 0.0,
    )


def crossfit_mucff(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    source_families: tuple[str, ...],
    config: MuCFFConfig | None = None,
) -> FusionPredictions:
    return _crossfit(
        oof_scores,
        labels,
        eval_scores,
        task_id,
        config or MuCFFConfig(),
        include_residual=True,
        source_families=source_families,
    )


def crossfit_aligned_control(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    config: MuCFFConfig | None = None,
) -> FusionPredictions:
    return _crossfit(
        oof_scores,
        labels,
        eval_scores,
        task_id,
        config or MuCFFConfig(),
        include_residual=False,
    )
