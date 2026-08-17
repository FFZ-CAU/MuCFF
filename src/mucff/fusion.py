"""Cross-fitted MuCFF estimator."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
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
    l2_regularization_c: float = 0.10
    max_iterations: int = 1800
    probability_epsilon: float = 1e-5
    routing_temperature: float = 0.20
    dual_linear_weight: float = 0.70
    xgb_estimators: int = 260
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.85
    xgb_column_subsample: float = 0.80
    xgb_l1_regularization: float = 0.05
    xgb_l2_regularization: float = 5.0
    xgb_threads: int = 2
    attention_dimension: int = 16
    attention_heads: int = 4
    attention_hidden_dimension: int = 32
    attention_dropout: float = 0.15
    attention_learning_rate: float = 1e-3
    attention_weight_decay: float = 1e-3
    attention_batch_size: int = 2048
    attention_max_epochs: int = 24
    attention_patience: int = 4
    attention_validation_fraction: float = 0.10
    attention_threads: int = 4


@dataclass(frozen=True)
class FusionPredictions:
    oof_probability: np.ndarray
    eval_probability: np.ndarray
    anchor_indices: tuple[int, ...] = ()
    residual_nonzero_fraction: float = 0.0


@dataclass(frozen=True)
class MuCFFPredictions(FusionPredictions):
    routed_oof_probability: np.ndarray | None = None
    routed_eval_probability: np.ndarray | None = None
    nonlinear_oof_probability: np.ndarray | None = None
    nonlinear_eval_probability: np.ndarray | None = None


def stable_seed(base: int, *parts: object) -> int:
    token = "|".join(map(str, parts)).encode("utf-8")
    return base + int(hashlib.sha256(token).hexdigest()[:8], 16) % 1_000_000


def make_l2_decision(config: MuCFFConfig):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.l2_regularization_c,
            max_iter=config.max_iterations,
            class_weight="balanced",
            random_state=config.model_seed,
        ),
    )


def make_xgboost_decision(config: MuCFFConfig):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "MuCFF requires xgboost. Install the package with `pip install -e .`."
        ) from exc
    return XGBClassifier(
        n_estimators=config.xgb_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_column_subsample,
        reg_alpha=config.xgb_l1_regularization,
        reg_lambda=config.xgb_l2_regularization,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cpu",
        n_jobs=config.xgb_threads,
        random_state=config.model_seed,
    )


def predict_probability(model, features: np.ndarray, epsilon: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = model.predict_proba(features)[:, 1]
    return clip_prob(values, epsilon)


def _splitter(task_id: str, config: MuCFFConfig) -> StratifiedKFold:
    return StratifiedKFold(
        config.outer_folds,
        shuffle=True,
        random_state=stable_seed(
            config.seed_base, task_id, "fusion_benchmark_common_cv"
        ),
    )


def _crossfit_aligned(
    estimator,
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    model_name: str,
    config: MuCFFConfig,
) -> FusionPredictions:
    train_state = aligned_state(oof_scores, config.probability_epsilon)
    eval_state = aligned_state(eval_scores, config.probability_epsilon)
    oof_probability = np.zeros(labels.size, dtype=np.float32)
    eval_probabilities: list[np.ndarray] = []
    for fold, (train_index, validation_index) in enumerate(
        _splitter(task_id, config).split(train_state, labels)
    ):
        model = clone(estimator)
        if hasattr(model, "random_state"):
            model.set_params(
                random_state=stable_seed(config.seed_base, task_id, model_name, fold)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_state[train_index], labels[train_index])
        oof_probability[validation_index] = predict_probability(
            model, train_state[validation_index], config.probability_epsilon
        )
        eval_probabilities.append(
            predict_probability(model, eval_state, config.probability_epsilon)
        )
    return FusionPredictions(
        oof_probability=clip_prob(
            oof_probability, config.probability_epsilon
        ).astype(np.float32),
        eval_probability=clip_prob(
            np.mean(eval_probabilities, axis=0), config.probability_epsilon
        ).astype(np.float32),
    )


def crossfit_aligned_control(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    config: MuCFFConfig | None = None,
) -> FusionPredictions:
    settings = config or MuCFFConfig()
    return _crossfit_aligned(
        make_l2_decision(settings),
        oof_scores,
        labels,
        eval_scores,
        task_id,
        "aligned_logistic_l2",
        settings,
    )


def crossfit_aligned_xgboost(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    config: MuCFFConfig | None = None,
) -> FusionPredictions:
    settings = config or MuCFFConfig()
    return _crossfit_aligned(
        make_xgboost_decision(settings),
        oof_scores,
        labels,
        eval_scores,
        task_id,
        "xgboost_stacking",
        settings,
    )


def crossfit_routed_l2(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    source_families: tuple[str, ...] | list[str],
    config: MuCFFConfig | None = None,
) -> FusionPredictions:
    settings = config or MuCFFConfig()
    oof_probability = np.zeros(labels.size, dtype=np.float32)
    eval_probabilities: list[np.ndarray] = []
    anchors: list[int] = []
    nonzero: list[float] = []
    aligned_dimension = aligned_state(
        oof_scores[:1], settings.probability_epsilon
    ).shape[1]
    for train_index, validation_index in _splitter(task_id, settings).split(
        oof_scores, labels
    ):
        routing = fit_routing_state(
            oof_scores[train_index],
            labels[train_index],
            source_families,
            settings.probability_epsilon,
        )
        train_state = mucff_state(
            oof_scores[train_index],
            routing,
            settings.routing_temperature,
            settings.probability_epsilon,
        )
        validation_state = mucff_state(
            oof_scores[validation_index],
            routing,
            settings.routing_temperature,
            settings.probability_epsilon,
        )
        eval_state = mucff_state(
            eval_scores,
            routing,
            settings.routing_temperature,
            settings.probability_epsilon,
        )
        model = clone(make_l2_decision(settings))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_state, labels[train_index])
        oof_probability[validation_index] = predict_probability(
            model, validation_state, settings.probability_epsilon
        )
        eval_probabilities.append(
            predict_probability(model, eval_state, settings.probability_epsilon)
        )
        anchors.append(routing.anchor_index)
        coefficients = model.named_steps["logisticregression"].coef_[0]
        nonzero.append(
            float(np.mean(np.abs(coefficients[aligned_dimension:]) > 1e-9))
        )
    return FusionPredictions(
        oof_probability=clip_prob(
            oof_probability, settings.probability_epsilon
        ).astype(np.float32),
        eval_probability=clip_prob(
            np.mean(eval_probabilities, axis=0), settings.probability_epsilon
        ).astype(np.float32),
        anchor_indices=tuple(anchors),
        residual_nonzero_fraction=float(np.mean(nonzero)),
    )


def logit_blend(
    linear_probability: np.ndarray,
    nonlinear_probability: np.ndarray,
    linear_weight: float,
    epsilon: float = 1e-5,
) -> np.ndarray:
    if not 0.0 <= linear_weight <= 1.0:
        raise ValueError("The linear-path weight must be in [0, 1].")
    linear = clip_prob(linear_probability, epsilon)
    nonlinear = clip_prob(nonlinear_probability, epsilon)
    return clip_prob(
        expit(
            linear_weight * logit(linear)
            + (1.0 - linear_weight) * logit(nonlinear)
        ),
        epsilon,
    ).astype(np.float32)


def crossfit_mucff(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    source_families: tuple[str, ...] | list[str],
    config: MuCFFConfig | None = None,
) -> MuCFFPredictions:
    settings = config or MuCFFConfig()
    routed = crossfit_routed_l2(
        oof_scores,
        labels,
        eval_scores,
        task_id,
        source_families,
        settings,
    )
    nonlinear = crossfit_aligned_xgboost(
        oof_scores, labels, eval_scores, task_id, settings
    )
    return MuCFFPredictions(
        oof_probability=logit_blend(
            routed.oof_probability,
            nonlinear.oof_probability,
            settings.dual_linear_weight,
            settings.probability_epsilon,
        ),
        eval_probability=logit_blend(
            routed.eval_probability,
            nonlinear.eval_probability,
            settings.dual_linear_weight,
            settings.probability_epsilon,
        ),
        anchor_indices=routed.anchor_indices,
        residual_nonzero_fraction=routed.residual_nonzero_fraction,
        routed_oof_probability=routed.oof_probability,
        routed_eval_probability=routed.eval_probability,
        nonlinear_oof_probability=nonlinear.oof_probability,
        nonlinear_eval_probability=nonlinear.eval_probability,
    )
