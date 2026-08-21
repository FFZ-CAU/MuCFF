"""Extended learned-fusion baselines used in the framework comparison."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .fusion import MuCFFConfig, crossfit_mucff, stable_seed
from .baselines import crossfit_logistic_control
from .fusion import crossfit_aligned_control
from .representation import aligned_state, clip_prob, logit, rank_columns, select_anchor


def family_context(scores: np.ndarray, families: tuple[str, ...], anchor_index: int) -> np.ndarray:
    probabilities = clip_prob(scores)
    family_order = tuple(dict.fromkeys(families))
    prototypes = []
    deviations = []
    for family in family_order:
        indices = [index for index, value in enumerate(families) if value == family]
        block = probabilities[:, indices]
        mean = block.mean(axis=1, keepdims=True)
        prototypes.extend([mean[:, 0], block.std(axis=1), block.min(axis=1), block.max(axis=1)])
        deviations.append(block - mean)
    family_means = np.column_stack(
        [
            probabilities[:, [index for index, value in enumerate(families) if value == family]].mean(axis=1)
            for family in family_order
        ]
    )
    family_logits = logit(family_means)
    family_ranks = rank_columns(family_means)
    products = []
    differences = []
    for left in range(len(family_order)):
        for right in range(left + 1, len(family_order)):
            products.append((family_logits[:, left] * family_logits[:, right])[:, None])
            differences.append((family_means[:, left] - family_means[:, right])[:, None])
    interactions = np.hstack(products + differences) if products else np.empty((len(scores), 0))
    anchor_probability = probabilities[:, anchor_index : anchor_index + 1]
    score_logits = logit(probabilities)
    anchor_logit = score_logits[:, anchor_index : anchor_index + 1]
    residual = np.hstack(
        [
            probabilities - anchor_probability,
            score_logits - anchor_logit,
            np.abs(probabilities - anchor_probability),
            np.abs(score_logits - anchor_logit),
            2.0 * np.abs(probabilities - 0.5),
        ]
    )
    return np.hstack(
        [
            aligned_state(probabilities),
            residual,
            np.column_stack(prototypes),
            np.hstack(deviations),
            family_means,
            family_logits,
            family_ranks,
            interactions,
        ]
    ).astype(np.float32)


def _predict(model, features: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(model, "predict_proba"):
            values = model.predict_proba(features)[:, 1]
        else:
            values = expit(model.decision_function(features))
    return clip_prob(values).astype(np.float32)


def _crossfit(
    estimator,
    train_state: np.ndarray,
    labels: np.ndarray,
    eval_state: np.ndarray,
    task_id: str,
    method: str,
    config: MuCFFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(
        config.outer_folds,
        shuffle=True,
        random_state=stable_seed(config.seed_base, task_id, "fusion_benchmark_common_cv"),
    )
    oof = np.zeros(labels.size, dtype=np.float32)
    evaluation = []
    for fold, (train_index, validation_index) in enumerate(splitter.split(train_state, labels)):
        model = clone(estimator)
        parameters = model.get_params(deep=False)
        if "random_state" in parameters:
            model.set_params(random_state=stable_seed(config.seed_base, task_id, method, fold))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_state[train_index], labels[train_index])
        oof[validation_index] = _predict(model, train_state[validation_index])
        evaluation.append(_predict(model, eval_state))
    return clip_prob(oof).astype(np.float32), clip_prob(np.mean(evaluation, axis=0)).astype(np.float32)


def _boosted_estimators(seed: int, device: str) -> dict[str, tuple[object, str]]:
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError("Install the 'benchmark' optional dependency.") from error
    return {
        "xgboost_stacking": (
            XGBClassifier(
                n_estimators=260,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=5.0,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                device=device,
                n_jobs=2,
                random_state=seed,
            ),
            "aligned",
        ),
        "lightgbm_stacking": (
            LGBMClassifier(
                n_estimators=260,
                num_leaves=15,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=5.0,
                verbosity=-1,
                n_jobs=2,
                random_state=seed,
            ),
            "aligned",
        ),
    }


def learned_estimators(seed: int, source_count: int, device: str) -> dict[str, tuple[object, str]]:
    models: dict[str, tuple[object, str]] = {
        "full_context_logistic_l2": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.03, max_iter=1800, class_weight="balanced", random_state=seed),
            ),
            "context",
        ),
        "nystroem_rbf_stacking": (
            make_pipeline(
                StandardScaler(),
                Nystroem(
                    kernel="rbf",
                    gamma=1.0 / max(source_count, 1),
                    n_components=192,
                    random_state=seed,
                ),
                LogisticRegression(C=0.1, max_iter=1600, class_weight="balanced", random_state=seed),
            ),
            "raw",
        ),
        "histogram_gradient_boosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=2.0,
                random_state=seed,
            ),
            "aligned",
        ),
        "multilayer_perceptron": (
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 16),
                    alpha=0.01,
                    learning_rate_init=5e-4,
                    early_stopping=True,
                    validation_fraction=0.15,
                    max_iter=300,
                    random_state=seed,
                ),
            ),
            "aligned",
        ),
    }
    models.update(_boosted_estimators(seed, device))
    return models


def rank_average(predictions: list[np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [pd.Series(values).rank(method="average", pct=True).to_numpy(float) for values in predictions]
    ).mean(axis=1)


def run_extended_baselines(ledger, config: MuCFFConfig, device: str = "cpu") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from .attention_baselines import run_qkv_attention_baselines

    anchor = select_anchor(ledger.oof_scores, ledger.y_oof)
    states = {
        "raw": clip_prob(ledger.oof_scores).astype(np.float32),
        "aligned": aligned_state(ledger.oof_scores),
        "context": family_context(ledger.oof_scores, ledger.source_families, anchor),
    }
    eval_states = {
        "raw": clip_prob(ledger.eval_scores).astype(np.float32),
        "aligned": aligned_state(ledger.eval_scores),
        "context": family_context(ledger.eval_scores, ledger.source_families, anchor),
    }
    aligned_l2 = crossfit_logistic_control(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.task_id,
        "aligned",
        config,
    )
    sparse_aligned = crossfit_aligned_control(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.task_id,
        config,
    )
    predictions = {
        "aligned_score_logistic_l2": (
            aligned_l2.oof_probability,
            aligned_l2.eval_probability,
        ),
        "sparse_aligned_control": (
            sparse_aligned.oof_probability,
            sparse_aligned.eval_probability,
        ),
    }
    for method, (estimator, state_name) in learned_estimators(
        config.model_seed, ledger.n_sources, device
    ).items():
        predictions[method] = _crossfit(
            estimator,
            states[state_name],
            ledger.y_oof,
            eval_states[state_name],
            ledger.task_id,
            method,
            config,
        )
    robust_methods = tuple(predictions)
    predictions["super_learner_rank_mean"] = (
        rank_average([predictions[name][0] for name in robust_methods]),
        rank_average([predictions[name][1] for name in robust_methods]),
    )
    predictions.update(run_qkv_attention_baselines(ledger, config, device))
    mucff = crossfit_mucff(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.task_id,
        config,
    )
    predictions["mucff"] = (mucff.oof_probability, mucff.eval_probability)
    return predictions


def run_attention_benchmark(
    ledger, config: MuCFFConfig, device: str = "cpu"
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from .attention_baselines import run_qkv_attention_baselines

    predictions = run_qkv_attention_baselines(ledger, config, device)
    aligned = crossfit_aligned_control(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.task_id,
        config,
    )
    mucff = crossfit_mucff(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.task_id,
        config,
    )
    predictions["sparse_aligned_control"] = (
        aligned.oof_probability,
        aligned.eval_probability,
    )
    predictions["mucff"] = (mucff.oof_probability, mucff.eval_probability)
    return predictions
