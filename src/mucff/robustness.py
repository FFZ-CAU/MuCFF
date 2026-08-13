"""Source-availability and score-perturbation analyses for MuCFF."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .fusion import (
    MuCFFConfig,
    crossfit_mucff,
    make_sparse_decision,
    predict_probability,
    stable_seed,
)
from .ledger import EvidenceLedger
from .metrics import binary_metrics, select_mcc_threshold
from .representation import clip_prob, logit, mucff_state, rank_columns, select_anchor


SOURCE_GROUPS: dict[str, frozenset[str]] = {
    "engineered_descriptors": frozenset(
        {
            "Composition and FCGR",
            "Motif and position",
            "Physicochemical",
            "DNA shape",
        }
    ),
    "foundation_evidence": frozenset(
        {
            "DNABERT-2",
            "DNABERT-1",
            "Nucleotide Transformer",
            "Cross-foundation interaction",
            "Pre-classifier local fusion",
        }
    ),
    "sequence_grammar": frozenset(
        {
            "RC motif grammar",
            "RC biophysical grammar",
            "Position-aware biophysical grammar",
        }
    ),
    "derived_evidence": frozenset(
        {
            "Deep feature interaction",
            "Score-residual evidence",
            "Cross-fitted meta-evidence",
        }
    ),
}


@dataclass(frozen=True)
class StressCondition:
    name: str
    kind: str
    source_indices: tuple[int, ...] = ()
    scale: float = 0.0
    replicate: int = 0


def _group_indices(ledger: EvidenceLedger) -> dict[str, tuple[int, ...]]:
    groups: dict[str, tuple[int, ...]] = {}
    families = np.asarray(ledger.source_families)
    for name, members in SOURCE_GROUPS.items():
        indices = tuple(np.flatnonzero(np.isin(families, tuple(members))).tolist())
        if indices:
            groups[name] = indices
    return groups


def _strongest_source(ledger: EvidenceLedger, epsilon: float) -> int:
    return select_anchor(ledger.oof_scores, ledger.y_oof, epsilon)


def build_stress_conditions(
    ledger: EvidenceLedger,
    config: MuCFFConfig,
    repeats: int = 20,
) -> list[StressCondition]:
    groups = _group_indices(ledger)
    conditions = [StressCondition("complete_sources", "clean")]
    conditions.extend(
        StressCondition(f"missing_group:{name}", "missing", indices)
        for name, indices in groups.items()
    )
    strongest = _strongest_source(ledger, config.probability_epsilon)
    conditions.extend(
        [
            StressCondition("missing_strongest_source", "missing", (strongest,)),
            StressCondition("conflicting_strongest_source", "conflict", (strongest,)),
        ]
    )
    source_count = ledger.n_sources
    for fraction in (0.20, 0.40):
        missing_count = max(1, int(round(fraction * source_count)))
        for replicate in range(repeats):
            rng = np.random.default_rng(
                stable_seed(config.seed_base, ledger.task_id, "missing", fraction, replicate)
            )
            indices = tuple(sorted(rng.choice(source_count, missing_count, replace=False).tolist()))
            conditions.append(
                StressCondition(
                    f"random_missing_{int(100 * fraction)}pct",
                    "missing",
                    indices,
                    replicate=replicate + 1,
                )
            )
    for replicate in range(repeats):
        conditions.append(
            StressCondition("score_noise_sd_0.05", "noise", scale=0.05, replicate=replicate + 1)
        )
    return conditions


def _alter_scores(
    scores: np.ndarray,
    condition: StressCondition,
    neutral_values: np.ndarray,
    seed: int,
    epsilon: float,
) -> np.ndarray:
    if condition.kind == "clean":
        return scores
    altered = scores.copy()
    if condition.kind == "missing":
        altered[:, condition.source_indices] = neutral_values[list(condition.source_indices)]
    elif condition.kind == "conflict":
        altered[:, condition.source_indices] = 1.0 - altered[:, condition.source_indices]
    elif condition.kind == "noise":
        rng = np.random.default_rng(seed)
        altered += rng.normal(0.0, condition.scale, size=altered.shape)
    else:
        raise ValueError(f"Unknown stress condition: {condition.kind}")
    return np.clip(altered, epsilon, 1.0 - epsilon)


def _state_from_components(
    probabilities: np.ndarray,
    ranks: np.ndarray,
    logits: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    summaries = np.column_stack(
        [
            probabilities.mean(axis=1),
            probabilities.std(axis=1),
            probabilities.min(axis=1),
            probabilities.max(axis=1),
            np.ptp(probabilities, axis=1),
        ]
    )
    anchor_probability = probabilities[:, anchor_index : anchor_index + 1]
    anchor_rank = ranks[:, anchor_index : anchor_index + 1]
    anchor_logit = logits[:, anchor_index : anchor_index + 1]
    residual = np.hstack(
        [
            probabilities - anchor_probability,
            ranks - anchor_rank,
            np.clip(logits - anchor_logit, -8.0, 8.0),
            np.abs(probabilities - anchor_probability),
        ]
    )
    return np.hstack([probabilities, ranks, logits, summaries, residual]).astype(np.float32)


def _condition_state(
    scores: np.ndarray,
    condition: StressCondition,
    neutral_values: np.ndarray,
    anchor_index: int,
    seed: int,
    epsilon: float,
    base_components: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    base_probabilities, base_ranks, base_logits = base_components
    if condition.kind == "clean":
        return _state_from_components(
            base_probabilities, base_ranks, base_logits, anchor_index
        )
    if condition.kind == "noise":
        probabilities = _alter_scores(
            scores, condition, neutral_values, seed, epsilon
        )
        return _state_from_components(
            probabilities,
            rank_columns(probabilities),
            logit(probabilities, epsilon),
            anchor_index,
        )

    probabilities = base_probabilities.copy()
    ranks = base_ranks.copy()
    logits = base_logits.copy()
    indices = list(condition.source_indices)
    if condition.kind == "missing":
        probabilities[:, indices] = neutral_values[indices]
        ranks[:, indices] = (scores.shape[0] + 1.0) / (2.0 * scores.shape[0])
        logits[:, indices] = logit(neutral_values[indices], epsilon)
    elif condition.kind == "conflict":
        probabilities[:, indices] = 1.0 - probabilities[:, indices]
        ranks[:, indices] = (scores.shape[0] + 1.0) / scores.shape[0] - ranks[:, indices]
        logits[:, indices] = -logits[:, indices]
    else:
        raise ValueError(f"Unknown stress condition: {condition.kind}")
    return _state_from_components(probabilities, ranks, logits, anchor_index)


def _predict_stress_conditions(
    ledger: EvidenceLedger,
    config: MuCFFConfig,
    conditions: list[StressCondition],
) -> tuple[np.ndarray, dict[StressCondition, np.ndarray]]:
    splitter = StratifiedKFold(
        config.outer_folds,
        shuffle=True,
        random_state=stable_seed(
            config.seed_base, ledger.task_id, "fusion_benchmark_common_cv"
        ),
    )
    oof_probability = np.zeros(ledger.y_oof.size, dtype=np.float32)
    fold_predictions = {condition: [] for condition in conditions}
    eval_probabilities = clip_prob(ledger.eval_scores, config.probability_epsilon)
    eval_components = (
        eval_probabilities,
        rank_columns(eval_probabilities),
        logit(eval_probabilities, config.probability_epsilon),
    )

    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(ledger.oof_scores, ledger.y_oof)
    ):
        anchor_index = select_anchor(
            ledger.oof_scores[train_index],
            ledger.y_oof[train_index],
            config.probability_epsilon,
        )
        train_state = mucff_state(
            ledger.oof_scores[train_index], anchor_index, config.probability_epsilon
        )
        validation_state = mucff_state(
            ledger.oof_scores[validation_index], anchor_index, config.probability_epsilon
        )
        model = clone(make_sparse_decision(config))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_state, ledger.y_oof[train_index])
        oof_probability[validation_index] = predict_probability(
            model, validation_state, config.probability_epsilon
        )

        neutral_values = np.median(ledger.oof_scores[train_index], axis=0)
        for condition in conditions:
            state = _condition_state(
                ledger.eval_scores,
                condition,
                neutral_values,
                anchor_index,
                stable_seed(
                    config.seed_base,
                    ledger.task_id,
                    condition.name,
                    condition.replicate,
                    fold_index,
                ),
                config.probability_epsilon,
                eval_components,
            )
            fold_predictions[condition].append(
                predict_probability(model, state, config.probability_epsilon)
            )

    averaged = {
        condition: np.mean(predictions, axis=0)
        for condition, predictions in fold_predictions.items()
    }
    return oof_probability, averaged


def _metric_row(
    ledger: EvidenceLedger,
    scenario: str,
    mode: str,
    replicate: int,
    affected_count: int,
    scores: np.ndarray,
    threshold: float,
    reference: dict[str, float],
) -> dict[str, float | int | str]:
    metrics = binary_metrics(ledger.y_eval, scores, threshold)
    return {
        "task_id": ledger.task_id,
        "scenario": scenario,
        "mode": mode,
        "replicate": replicate,
        "source_count": ledger.n_sources,
        "affected_count": affected_count,
        **metrics,
        "delta_auc": metrics["auc"] - reference["auc"],
        "delta_auprc": metrics["auprc"] - reference["auprc"],
        "delta_mcc": metrics["mcc"] - reference["mcc"],
        "delta_accuracy": metrics["accuracy"] - reference["accuracy"],
    }


def run_robustness(
    ledger: EvidenceLedger,
    config: MuCFFConfig,
    threshold_settings: dict[str, float | int],
    repeats: int = 20,
) -> pd.DataFrame:
    conditions = build_stress_conditions(ledger, config, repeats)
    oof_probability, stress_predictions = _predict_stress_conditions(
        ledger, config, conditions
    )
    threshold = select_mcc_threshold(
        ledger.y_oof,
        oof_probability,
        float(threshold_settings["threshold_quantile_low"]),
        float(threshold_settings["threshold_quantile_high"]),
        int(threshold_settings["threshold_quantile_count"]),
    )
    clean_condition = conditions[0]
    reference = binary_metrics(
        ledger.y_eval, stress_predictions[clean_condition], threshold
    )
    rows = [
        _metric_row(
            ledger,
            condition.name,
            "deployment_shift",
            condition.replicate,
            len(condition.source_indices),
            scores,
            threshold,
            reference,
        )
        for condition, scores in stress_predictions.items()
    ]

    for group_name, removed_indices in _group_indices(ledger).items():
        retained = np.ones(ledger.n_sources, dtype=bool)
        retained[list(removed_indices)] = False
        prediction = crossfit_mucff(
            ledger.oof_scores[:, retained],
            ledger.y_oof,
            ledger.eval_scores[:, retained],
            ledger.task_id,
            config,
        )
        removed_threshold = select_mcc_threshold(
            ledger.y_oof,
            prediction.oof_probability,
            float(threshold_settings["threshold_quantile_low"]),
            float(threshold_settings["threshold_quantile_high"]),
            int(threshold_settings["threshold_quantile_count"]),
        )
        rows.append(
            _metric_row(
                ledger,
                f"refit_without_group:{group_name}",
                "source_ablation",
                0,
                len(removed_indices),
                prediction.eval_probability,
                removed_threshold,
                reference,
            )
        )
    return pd.DataFrame(rows)


def summarize_robustness(results: pd.DataFrame) -> pd.DataFrame:
    per_task = (
        results.groupby(["task_id", "scenario", "mode"], as_index=False)
        .agg(
            replicates=("replicate", "count"),
            affected_count=("affected_count", "mean"),
            auc=("auc", "mean"),
            auprc=("auprc", "mean"),
            mcc=("mcc", "mean"),
            accuracy=("accuracy", "mean"),
            delta_auc=("delta_auc", "mean"),
            delta_auprc=("delta_auprc", "mean"),
            delta_mcc=("delta_mcc", "mean"),
            delta_accuracy=("delta_accuracy", "mean"),
        )
    )
    return (
        per_task.groupby(["scenario", "mode"], as_index=False)
        .agg(
            tasks=("task_id", "nunique"),
            replicates_per_task=("replicates", "median"),
            mean_affected_sources=("affected_count", "mean"),
            mean_auc=("auc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_mcc=("mcc", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_delta_auc=("delta_auc", "mean"),
            mean_delta_auprc=("delta_auprc", "mean"),
            mean_delta_mcc=("delta_mcc", "mean"),
            mean_delta_accuracy=("delta_accuracy", "mean"),
            positive_or_tied_auc_tasks=("delta_auc", lambda x: int(np.sum(x >= 0.0))),
        )
        .sort_values(["mode", "mean_delta_auc"], ascending=[True, False])
    )
