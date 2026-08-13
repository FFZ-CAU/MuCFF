"""Cross-fitted source heads for engineered and precomputed representations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .fusion import stable_seed
from .representation import clip_prob


@dataclass(frozen=True)
class SourcePrediction:
    oof_probability: np.ndarray
    eval_probability: np.ndarray


def logistic_head(c_value: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            max_iter=2500,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        ),
    )


def extra_trees_head(seed: int, trees: int = 300, threads: int = 2):
    return ExtraTreesClassifier(
        n_estimators=trees,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=threads,
    )


def crossfit_source(
    train_features,
    labels: np.ndarray,
    eval_features,
    estimator,
    task_id: str,
    source_id: str,
    folds: int = 4,
    seed_base: int = 20260806,
) -> SourcePrediction:
    splitter = StratifiedKFold(
        folds,
        shuffle=True,
        random_state=stable_seed(seed_base, task_id, source_id, "source_cv"),
    )
    oof = np.zeros(labels.size, dtype=np.float32)
    evaluation = []
    for train_index, validation_index in splitter.split(np.zeros(labels.size), labels):
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_features[train_index], labels[train_index])
        oof[validation_index] = model.predict_proba(train_features[validation_index])[:, 1]
        evaluation.append(model.predict_proba(eval_features)[:, 1])
    final_model = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(train_features, labels)
    final_evaluation = final_model.predict_proba(eval_features)[:, 1]
    fold_evaluation = np.mean(evaluation, axis=0)
    return SourcePrediction(
        clip_prob(oof).astype(np.float32),
        clip_prob(0.5 * final_evaluation + 0.5 * fold_evaluation).astype(np.float32),
    )
