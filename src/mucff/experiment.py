"""Unified experiment runner for MuCFF and matched controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
        regularization_c=float(values["regularization_c"]),
        l1_ratio=float(values["l1_ratio"]),
        max_iterations=int(values["max_iterations"]),
        probability_epsilon=float(values["probability_epsilon"]),
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
    predictions["raw_score_logistic_l2"] = crossfit_logistic_control(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.task_id,
        "raw",
        config,
    )
    predictions["aligned_score_logistic_l2"] = crossfit_logistic_control(
        ledger.oof_scores,
        ledger.y_oof,
        ledger.eval_scores,
        ledger.task_id,
        "aligned",
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
        config,
    )
    predictions["sparse_aligned_control"] = BaselinePrediction(
        aligned.oof_probability, aligned.eval_probability
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


def compare_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    return compare_auc(metrics, reference_method="mucff")
