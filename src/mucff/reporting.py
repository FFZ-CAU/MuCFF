"""Publication-oriented plots from the released result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _style() -> None:
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


def _comparison_figure(reference: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    comparison = pd.read_csv(reference / "matched_framework_comparisons.csv")
    selected_names = [
        "xgboost_stacking",
        "raw_logistic_l2",
        "superlearner_rank_mean",
        "aligned_elastic_net",
        "anchor_query_qkv",
        "aligned_logistic_l2",
        "OOF-selected individual evidence channel",
        "choquet_pruned_oof",
        "probability_mean",
    ]
    selected = comparison.loc[comparison.comparator.isin(selected_names), ["comparator", "mean_auc"]]
    selected = pd.concat(
        [
            selected,
            pd.DataFrame([{"comparator": "MuCFF", "mean_auc": 0.9298014876255334}]),
        ],
        ignore_index=True,
    ).sort_values("mean_auc")
    labels = {
        "xgboost_stacking": "XGBoost stacking",
        "raw_logistic_l2": "Raw-score logistic",
        "superlearner_rank_mean": "Super Learner",
        "aligned_elastic_net": "Aligned elastic net",
        "anchor_query_qkv": "Anchor-query QKV",
        "aligned_logistic_l2": "Aligned logistic",
        "OOF-selected individual evidence channel": "Best OOF-selected channel",
        "choquet_pruned_oof": "Choquet integral",
        "probability_mean": "Probability mean",
        "MuCFF": "MuCFF",
    }
    figure, axis = plt.subplots(figsize=(6.9, 3.6), constrained_layout=True)
    positions = np.arange(len(selected))
    colors = ["#B6454D" if name == "MuCFF" else "#247B80" for name in selected.comparator]
    baseline = min(0.79, float(selected.mean_auc.min()) - 0.005)
    axis.hlines(positions, baseline, selected.mean_auc, color="#D5DADF", linewidth=1.1)
    axis.scatter(selected.mean_auc, positions, c=colors, s=34, edgecolor="white", linewidth=0.5)
    axis.set_yticks(positions, [labels[name] for name in selected.comparator])
    axis.set_xlim(baseline, 0.935)
    axis.set_xlabel("Mean AUC across ten tasks")
    axis.grid(axis="x", color="#E2E5E8", linewidth=0.6)
    _save(figure, output, "matched_fusion_comparison")
    plt.close(figure)


def _task_figure(reference: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    taskwise = pd.read_csv(reference / "framework_task_metrics.csv").sort_values("mucff_auc")
    figure, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    positions = np.arange(len(taskwise))
    axis.plot(taskwise.individual_channel_auc, positions, "o", color="#8A949D", label="OOF-selected channel")
    axis.plot(taskwise.aligned_l2_auc, positions, "D", color="#356BA4", label="Aligned logistic")
    axis.plot(taskwise.mucff_auc, positions, "o", color="#B6454D", label="MuCFF")
    for position, row in enumerate(taskwise.itertuples(index=False)):
        axis.hlines(
            position,
            min(row.individual_channel_auc, row.aligned_l2_auc, row.mucff_auc),
            max(row.individual_channel_auc, row.aligned_l2_auc, row.mucff_auc),
            color="#D5DADF",
            linewidth=1.0,
            zorder=0,
        )
    axis.set_yticks(positions, taskwise.task_id)
    axis.set_xlabel("Evaluation AUC")
    axis.grid(axis="x", color="#E2E5E8", linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    _save(figure, output, "taskwise_framework_comparison")
    plt.close(figure)


def _growth_figure(reference: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    growth = pd.read_csv(reference / "evidence_growth_summary.csv").sort_values("order")
    labels = {
        "base_evidence": "Base evidence",
        "foundation_interactions": "+ foundation interactions",
        "local_foundation_fusion": "+ local fusion",
        "rc_grammar_1": "+ RC grammar 1",
        "rc_grammar_2": "+ RC grammar 2",
        "rc_grammar_3": "+ RC grammar 3",
        "biophysical_grammar": "+ biophysical grammar",
        "position_aware_grammar": "+ position-aware grammar",
        "reliability_routed_l2": "+ complementary routing",
        "mucff_dual_decision": "+ dual decision",
    }
    figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    positions = np.arange(len(growth))
    colors = ["#B6454D" if stage == "mucff_dual_decision" else "#247B80" for stage in growth.stage]
    baseline = float(growth.mean_auc.min()) - 0.002
    axis.hlines(positions, baseline, growth.mean_auc, color="#D5DADF", linewidth=1.1)
    axis.scatter(growth.mean_auc, positions, c=colors, s=34, edgecolor="white", linewidth=0.5)
    axis.set_yticks(positions, [labels[stage] for stage in growth.stage])
    axis.set_xlim(baseline, float(growth.mean_auc.max()) + 0.002)
    axis.set_xlabel("Mean AUC across ten tasks")
    axis.grid(axis="x", color="#E2E5E8", linewidth=0.6)
    axis.invert_yaxis()
    _save(figure, output, "evidence_bank_growth")
    plt.close(figure)


def make_result_figures(reference_root: str | Path, output_root: str | Path) -> None:
    _style()
    reference = Path(reference_root)
    output = Path(output_root)
    _comparison_figure(reference, output)
    _task_figure(reference, output)
    _growth_figure(reference, output)
