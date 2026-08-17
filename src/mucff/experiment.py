"""Unified experiment runner for MuCFF and matched controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

from .baselines import BaselinePrediction, crossfit_logistic_control, fixed_fusion_baselines
from .fusion import MuCFFConfig, crossfit_aligned_control, crossfit_mucff
from .ledger import EvidenceLedger
from .metrics import binary_metrics, select_mcc_threshold
from .statistics import compare_auc


def load_config(path: str | Path) -> MuCFFConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return MuCFFConfig(
        outer_folds=int(values["outer_folds"]),
        seed_base=int(values["seed_base"]),
        model_seed=int(values["model_seed"]),
        l2_regularization_c=float(values["l2_regularization_c"]),
        max_iterations=int(values["max_iterations"]),
        probability_epsilon=float(values["probability_epsilon"]),
        routing_temperature=float(values.get("routing_temperature", 0.20)),
        dual_linear_weight=float(values.get("dual_linear_weight", 0.70)),
        xgb_estimators=int(values.get("xgb_estimators", 260)),
        xgb_max_depth=int(values.get("xgb_max_depth", 3)),
        xgb_learning_rate=float(values.get("xgb_learning_rate", 0.03)),
        xgb_subsample=float(values.get("xgb_subsample", 0.85)),
        xgb_column_subsample=float(values.get("xgb_column_subsample", 0.80)),
        xgb_l1_regularization=float(values.get("xgb_l1_regularization", 0.05)),
        xgb_l2_regularization=float(values.get("xgb_l2_regularization", 5.0)),
        xgb_threads=int(values.get("xgb_threads", 2)),
        attention_dimension=int(values.get("attention_dimension", 16)),
        attention_heads=int(values.get("attention_heads", 4)),
        attention_hidden_dimension=int(values.get("attention_hidden_dimension", 32)),
        attention_dropout=float(values.get("attention_dropout", 0.15)),
        attention_learning_rate=float(values.get("attention_learning_rate", 1e-3)),
        attention_weight_decay=float(values.get("attention_weight_decay", 1e-3)),
        attention_batch_size=int(values.get("attention_batch_size", 2048)),
        attention_max_epochs=int(values.get("attention_max_epochs", 24)),
        attention_patience=int(values.get("attention_patience", 4)),
        attention_validation_fraction=float(
            values.get("attention_validation_fraction", 0.10)
        ),
        attention_threads=int(values.get("attention_threads", 4)),
    )


def run_task(
    ledger: EvidenceLedger,
    config: MuCFFConfig,
    threshold_settings: dict[str, float | int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    predictions = fixed_fusion_baselines(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.source_families,
        ledger.source_ids,
    )
    predictions["raw_logistic_l2"] = crossfit_logistic_control(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.task_id,
        "raw",
        config,
    )
    aligned = crossfit_aligned_control(
        ledger.oof_scores, ledger.y_oof, ledger.eval_scores, ledger.task_id, config
    )
    mucff_result = crossfit_mucff(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.task_id,
        ledger.source_families,
        config,
    )
    predictions["aligned_logistic_l2"] = BaselinePrediction(
        aligned.oof_probability, aligned.eval_probability
    )
    predictions["routed_logistic_l2"] = BaselinePrediction(
        mucff_result.routed_oof_probability, mucff_result.routed_eval_probability
    )
    predictions["aligned_xgboost"] = BaselinePrediction(
        mucff_result.nonlinear_oof_probability,
        mucff_result.nonlinear_eval_probability,
    )
    predictions["mucff"] = BaselinePrediction(
        mucff_result.oof_probability, mucff_result.eval_probability
    )

    rows = []
    arrays: dict[str, np.ndarray] = {
        "y_oof": ledger.y_oof,
        "y_eval": ledger.y_eval,
    }
    for method, prediction in predictions.items():
        threshold = select_mcc_threshold(
            ledger.y_oof,
            prediction.oof_probability,
            float(threshold_settings["threshold_quantile_low"]),
            float(threshold_settings["threshold_quantile_high"]),
            int(threshold_settings["threshold_quantile_count"]),
        )
        row = {
            "task_id": ledger.task_id,
            "method": method,
            "n_sources": ledger.n_sources,
        }
        row.update(binary_metrics(ledger.y_eval, prediction.eval_probability, threshold))
        rows.append(row)
        arrays[f"{method}_oof"] = prediction.oof_probability.astype(np.float32)
        arrays[f"{method}_eval"] = prediction.eval_probability.astype(np.float32)
    return pd.DataFrame(rows), arrays


def write_task_results(
    output_root: str | Path,
    ledger: EvidenceLedger,
    metrics: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> None:
    directory = Path(output_root) / ledger.task_id
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    np.savez_compressed(directory / "predictions.npz", **arrays)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "auc",
        "auprc",
        "mcc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    ]
    summary = metrics.groupby("method", as_index=False)[numeric].mean()
    mucff_auc = float(summary.loc[summary.method.eq("mucff"), "auc"].iloc[0])
    summary["mucff_minus_method_auc"] = mucff_auc - summary["auc"]
    return summary.sort_values("auc", ascending=False)


def _pooled_metric_row(
    labels: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def aggregate_evaluation_tasks(
    output_root: str | Path,
    metrics: pd.DataFrame,
    threshold_settings: dict[str, float | int],
) -> pd.DataFrame:
    """Pool the ten Rice outer tests while preserving all other task rows."""
    rice_prefix = "snnrice6ma_rice_chen__fold"
    rice_rows = metrics[metrics["task_id"].astype(str).str.startswith(rice_prefix)]
    if rice_rows.empty:
        return metrics.copy()
    retained = metrics[~metrics.index.isin(rice_rows.index)].copy()
    pooled_rows: list[dict[str, object]] = []
    for method in sorted(rice_rows["method"].unique()):
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        source_counts: list[int] = []
        for task_id in sorted(rice_rows["task_id"].unique()):
            with np.load(Path(output_root) / task_id / "predictions.npz") as stored:
                y_oof = stored["y_oof"].astype(int)
                y_eval = stored["y_eval"].astype(int)
                oof = stored[f"{method}_oof"].astype(float)
                evaluation = stored[f"{method}_eval"].astype(float)
            threshold = select_mcc_threshold(
                y_oof,
                oof,
                float(threshold_settings["threshold_quantile_low"]),
                float(threshold_settings["threshold_quantile_high"]),
                int(threshold_settings["threshold_quantile_count"]),
            )
            labels.append(y_eval)
            probabilities.append(evaluation)
            predictions.append(evaluation >= threshold)
            source_counts.append(
                int(
                    rice_rows.loc[
                        rice_rows["task_id"].eq(task_id)
                        & rice_rows["method"].eq(method),
                        "n_sources",
                    ].iloc[0]
                )
            )
        row: dict[str, object] = {
            "task_id": "snnrice6ma_rice_chen",
            "method": method,
            "n_sources": int(round(np.mean(source_counts))),
        }
        row.update(
            _pooled_metric_row(
                np.concatenate(labels),
                np.concatenate(probabilities),
                np.concatenate(predictions),
            )
        )
        pooled_rows.append(row)
    return pd.concat([retained, pd.DataFrame(pooled_rows)], ignore_index=True)


def compare_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    return compare_auc(metrics, reference_method="mucff")
