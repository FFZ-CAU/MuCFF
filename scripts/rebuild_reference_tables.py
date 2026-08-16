"""Rebuild derived reference tables from the canonical released results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mucff.statistics import _paired_summary


OPERATOR_LABELS = {
    "choquet_integral": ("Choquet integral (pruned)", "capacity-based fusion"),
    "correlation_pruned_weighted_pool": (
        "Correlation-pruned weighted pool",
        "redundancy-pruned pooling",
    ),
    "family_best_weighted_pool": ("Family-best weighted pool", "redundancy-pruned pooling"),
    "reliability_attention": ("Reliability attention", "attention fusion"),
    "oof_skill_linear_pool": ("OOF-skill linear pool", "linear opinion pool"),
    "oof_skill_logarithmic_pool": (
        "OOF-skill logarithmic pool",
        "logarithmic opinion pool",
    ),
    "probability_mean": ("Probability mean", "classical pooling"),
    "probability_median": ("Probability median", "classical pooling"),
    "rank_mean": ("Rank mean", "rank fusion"),
    "majority_vote": ("Majority vote", "voting"),
    "dempster_shafer_pool": ("Dempster-Shafer pool", "belief-function fusion"),
}


def _operator_table(metrics: pd.DataFrame) -> pd.DataFrame:
    taskwise = metrics.pivot(index="task_id", columns="method", values="auc")
    rows = []
    for method, (label, detail) in OPERATOR_LABELS.items():
        selected = metrics.loc[metrics.method.eq(method)]
        differences = (taskwise["mucff"] - taskwise[method]).to_numpy()
        paired = _paired_summary(differences)
        rows.append(
            {
                "recipe_label": label,
                "detail": detail,
                "mean_auc": selected.auc.mean(),
                "mean_auprc": selected.auprc.mean(),
                "mean_mcc": selected.mcc.mean(),
                "mean_accuracy": selected.accuracy.mean(),
                "mucff_minus_method_auc": paired["mean_auc_difference"],
                "mucff_wtl_vs_method": (
                    f'{paired["wins"]}/{paired["ties"]}/{paired["losses"]}'
                ),
                "mucff_vs_method_p": paired["wilcoxon_one_sided_p"],
            }
        )
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False)


def _score_representation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    taskwise = metrics.pivot(index="task_id", columns="method", values="auc")
    raw = metrics.loc[metrics.method.eq("raw_score_logistic_l2")]
    aligned = metrics.loc[metrics.method.eq("aligned_score_logistic_l2")]
    paired = _paired_summary(
        (taskwise["aligned_score_logistic_l2"] - taskwise["raw_score_logistic_l2"]).to_numpy()
    )
    return pd.DataFrame(
        [
            {
                "score_representation": "Raw source probabilities",
                "mean_auc": raw.auc.mean(),
                "mean_auprc": raw.auprc.mean(),
                "mean_mcc": raw.mcc.mean(),
                "delta_auc_vs_raw": 0.0,
                "ci95_low_vs_raw": 0.0,
                "ci95_high_vs_raw": 0.0,
                "wtl_vs_raw": "0/10/0",
                "p_vs_raw": 1.0,
            },
            {
                "score_representation": (
                    "Probability plus rank plus logit plus global summaries"
                ),
                "mean_auc": aligned.auc.mean(),
                "mean_auprc": aligned.auprc.mean(),
                "mean_mcc": aligned.mcc.mean(),
                "delta_auc_vs_raw": paired["mean_auc_difference"],
                "ci95_low_vs_raw": paired["ci95_low"],
                "ci95_high_vs_raw": paired["ci95_high"],
                "wtl_vs_raw": f'{paired["wins"]}/{paired["ties"]}/{paired["losses"]}',
                "p_vs_raw": paired["wilcoxon_one_sided_p"],
            },
        ]
    )


def _taskwise_decomposition(reference: Path, data_root: Path) -> pd.DataFrame:
    growth = pd.read_csv(reference / "evidence_growth_task_metrics.csv")
    table = growth.pivot(index="task_id", columns="stage", values="auc")
    output = pd.DataFrame(
        {
            "task_id": table.index,
            "base_auc": table["foundation_local_bank"].to_numpy(),
            "aligned_auc": table["plus_positional_biophysical_grammar"].to_numpy(),
            "mucff_auc": table["complete_mucff"].to_numpy(),
        }
    )
    output["evidence_gain"] = output.aligned_auc - output.base_auc
    output["residual_gain"] = output.mucff_auc - output.aligned_auc
    labels = pd.read_csv(data_root / "dataset_manifest.csv")[["task_id", "task_label"]]
    return labels.merge(output, on="task_id", how="inner")


def _clean_source_summary(reference: Path, data_root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(reference / "source_performance_summary.csv")
    metadata = pd.read_csv(data_root / "source_metadata.csv")
    numeric = [
        "n_tasks",
        "mean_oof_auc",
        "mean_evaluation_auc",
        "mean_auprc",
        "mean_mcc",
        "mean_accuracy",
    ]
    return metadata.merge(metrics[["source_id", *numeric]], on="source_id", how="left")


def rebuild(reference: Path, data_root: Path) -> None:
    metrics = pd.read_csv(reference / "reference_task_metrics.csv")
    _operator_table(metrics).to_csv(reference / "fusion_operator_comparison.csv", index=False)
    _score_representation_table(metrics).to_csv(
        reference / "score_representation_analysis.csv", index=False
    )
    mechanism = pd.read_csv(reference / "mechanism_ablation_summary.csv")
    mechanism.to_csv(reference / "fusion_head_sensitivity.csv", index=False)
    _taskwise_decomposition(reference, data_root).to_csv(
        reference / "taskwise_mechanism_decomposition.csv", index=False
    )
    _clean_source_summary(reference, data_root).to_csv(
        reference / "source_performance_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("results/reference"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    rebuild(args.reference, args.data)


if __name__ == "__main__":
    main()
