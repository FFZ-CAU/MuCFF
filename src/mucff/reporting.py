"""Publication-oriented result figures from released tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            output / f"{stem}.{extension}",
            dpi=400 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
        )


def make_result_figures(reference_root: str | Path, output_root: str | Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _style()
    reference = Path(reference_root)
    output = Path(output_root)

    comparison = pd.read_csv(reference / "framework_method_summary.csv")
    selected = comparison[
        comparison.method.isin(
            [
                "best_standalone_source",
                "probability_mean",
                "reliability_attention",
                "choquet_integral",
                "raw_score_logistic_l2",
                "xgboost_stacking",
                "super_learner_rank_mean",
                "aligned_score_logistic_l2",
                "sparse_aligned_control",
                "mucff",
            ]
        )
    ].sort_values("auc")
    figure, axis = plt.subplots(figsize=(6.9, 3.7), constrained_layout=True)
    positions = np.arange(len(selected))
    colors = ["#C74B50" if name == "MuCFF" else "#287D82" for name in selected.method_display]
    axis.hlines(positions, 0.90, selected.auc, color="#D5DADF", linewidth=1.2)
    axis.scatter(selected.auc, positions, c=colors, s=34, edgecolor="white", linewidth=0.5)
    axis.set_yticks(positions, selected.method_display)
    axis.set_xlim(0.90, 0.95)
    axis.set_xlabel("Mean AUC across ten tasks")
    axis.grid(axis="x", color="#E2E5E8", linewidth=0.6)
    _save(figure, output, "framework_comparison")
    plt.close(figure)

    task_metrics = pd.read_csv(reference / "framework_task_metrics.csv")
    taskwise = task_metrics[
        task_metrics.method.isin(["best_standalone_source", "sparse_aligned_control", "mucff"])
    ].pivot(index="task_id", columns="method", values="auc").reset_index()
    taskwise = taskwise.sort_values("mucff")
    figure, axis = plt.subplots(figsize=(7.1, 4.0), constrained_layout=True)
    positions = np.arange(len(taskwise))
    axis.plot(taskwise.best_standalone_source, positions, "o", color="#8A949D", label="Best standalone")
    axis.plot(taskwise.sparse_aligned_control, positions, "D", color="#3E6FAF", label="Aligned sparse decision")
    axis.plot(taskwise.mucff, positions, "o", color="#C74B50", label="MuCFF")
    for position, row in enumerate(taskwise.itertuples(index=False)):
        axis.hlines(position, row.best_standalone_source, row.mucff, color="#D5DADF", linewidth=1.0, zorder=0)
    axis.set_yticks(positions, taskwise.task_id.str.replace("_", " "))
    axis.set_xlabel("Evaluation AUC")
    axis.grid(axis="x", color="#E2E5E8", linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    _save(figure, output, "mechanism_decomposition")
    plt.close(figure)

    expansion = pd.read_csv(reference / "evidence_growth_summary.csv")
    figure, axis = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    positions = np.arange(len(expansion))
    labels = expansion.stage.str.replace("_", " ")
    baseline = float(expansion.mean_auc.min()) - 0.0015
    axis.hlines(positions, baseline, expansion.mean_auc, color="#D5DADF", linewidth=1.2)
    axis.scatter(
        expansion.mean_auc,
        positions,
        color=["#C74B50" if index == len(expansion) - 1 else "#287D82" for index in positions],
        s=38,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Mean AUC across ten tasks")
    axis.set_xlim(baseline, float(expansion.mean_auc.max()) + 0.0015)
    axis.grid(axis="x", color="#E2E5E8", linewidth=0.6)
    axis.invert_yaxis()
    _save(figure, output, "evidence_construction")
    plt.close(figure)
