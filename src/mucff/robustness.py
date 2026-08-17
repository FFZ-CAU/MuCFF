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
    logit_blend,
    make_l2_decision,
    make_xgboost_decision,
    predict_probability,
    stable_seed,
)
from .ledger import EvidenceLedger
from .metrics import binary_metrics, select_mcc_threshold
from .representation import aligned_state, clip_prob, fit_routing_state, mucff_state, select_anchor


SOURCE_GROUPS: dict[str, frozenset[str]] = {
    "engineered_descriptors": frozenset(
        {
            "composition",
            "motif_position",
            "physicochemical",
            "dna_shape",
        }
    ),
    "foundation_evidence": frozenset(
        {
            "foundation_db1",
            "foundation_db2",
            "foundation_nt",
            "foundation_cross",
            "foundation_residual",
        }
    ),
    "sequence_grammar": frozenset(
        {
            "sequence_grammar_rc",
            "sequence_grammar_biophysical",
            "sequence_grammar_position",
        }
    ),
    "derived_evidence": frozenset(
        {
            "derived_anchor",
            "preclassifier_local",
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
                stable_seed(
                    config.seed_base,
                    ledger.task_id,
                    "deployment_missing",
                    fraction,
                    replicate + 1,
                )
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


def _condition_state(
    scores: np.ndarray,
    condition: StressCondition,
    neutral_values: np.ndarray,
    routing,
    seed: int,
    config: MuCFFConfig,
) -> np.ndarray:
    probabilities = _alter_scores(
        scores,
        condition,
        neutral_values,
        seed,
        config.probability_epsilon,
    )
    return mucff_state(
        probabilities,
        routing,
        config.routing_temperature,
        config.probability_epsilon,
    )


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
    routed_oof = np.zeros(ledger.y_oof.size, dtype=np.float32)
    nonlinear_oof = np.zeros(ledger.y_oof.size, dtype=np.float32)
    routed_predictions = {condition: [] for condition in conditions}
    nonlinear_predictions = {condition: [] for condition in conditions}
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(ledger.oof_scores, ledger.y_oof)
    ):
        routing = fit_routing_state(
            ledger.oof_scores[train_index],
            ledger.y_oof[train_index],
            ledger.source_families,
            config.probability_epsilon,
        )
        train_state = mucff_state(
            ledger.oof_scores[train_index],
            routing,
            config.routing_temperature,
            config.probability_epsilon,
        )
        validation_state = mucff_state(
            ledger.oof_scores[validation_index],
            routing,
            config.routing_temperature,
            config.probability_epsilon,
        )
        routed_model = clone(make_l2_decision(config))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            routed_model.fit(train_state, ledger.y_oof[train_index])
        routed_oof[validation_index] = predict_probability(
            routed_model, validation_state, config.probability_epsilon
        )
        nonlinear_model = clone(make_xgboost_decision(config))
        nonlinear_model.set_params(
            random_state=stable_seed(
                config.seed_base, ledger.task_id, "xgboost_stacking", fold_index
            )
        )
        nonlinear_model.fit(
            aligned_state(
                ledger.oof_scores[train_index], config.probability_epsilon
            ),
            ledger.y_oof[train_index],
        )
        nonlinear_oof[validation_index] = predict_probability(
            nonlinear_model,
            aligned_state(
                ledger.oof_scores[validation_index], config.probability_epsilon
            ),
            config.probability_epsilon,
        )

        neutral_values = np.median(ledger.oof_scores[train_index], axis=0)
        for condition in conditions:
            state = _condition_state(
                ledger.eval_scores,
                condition,
                neutral_values,
                routing,
                stable_seed(
                    config.seed_base,
                    ledger.task_id,
                    "deployment_noise",
                    fold_index,
                    condition.replicate,
                ),
                config,
            )
            altered = _alter_scores(
                ledger.eval_scores,
                condition,
                neutral_values,
                stable_seed(
                    config.seed_base,
                    ledger.task_id,
                    "deployment_noise",
                    fold_index,
                    condition.replicate,
                ),
                config.probability_epsilon,
            )
            routed_predictions[condition].append(
                predict_probability(routed_model, state, config.probability_epsilon)
            )
            nonlinear_predictions[condition].append(
                predict_probability(
                    nonlinear_model,
                    aligned_state(altered, config.probability_epsilon),
                    config.probability_epsilon,
                )
            )

    averaged = {
        condition: logit_blend(
            np.mean(routed_predictions[condition], axis=0),
            np.mean(nonlinear_predictions[condition], axis=0),
            config.dual_linear_weight,
            config.probability_epsilon,
        )
        for condition in conditions
    }
    return (
        logit_blend(
            routed_oof,
            nonlinear_oof,
            config.dual_linear_weight,
            config.probability_epsilon,
        ),
        averaged,
    )


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
            tuple(np.asarray(ledger.source_families)[retained]),
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
