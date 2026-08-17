#!/usr/bin/env python
"""Export the compact reference archive from a completed primary run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mucff.statistics import _paired_summary


METHODS = (
    "aligned_logistic_l2",
    "routed_logistic_l2",
    "aligned_xgboost",
    "mucff",
)
METRICS = (
    "auc",
    "auprc",
    "mcc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("outputs/main_experiment"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reference_export"))
    args = parser.parse_args()

    metrics = pd.read_csv(args.run / "task_metrics.csv")
    selected = metrics.loc[metrics.method.isin(METHODS)].copy()
    if selected.task_id.nunique() != 10:
        raise ValueError("The primary export requires ten aggregated tasks.")
    args.output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output / "selected_framework_metrics.csv", index=False)

    pivot = selected.pivot(index="task_id", columns="method")
    rows = []
    for metric in METRICS:
        difference = (
            pivot[metric]["mucff"] - pivot[metric]["aligned_logistic_l2"]
        ).to_numpy()
        paired = _paired_summary(difference)
        paired.pop("mean_auc_difference")
        rows.append(
            {
                "metric": metric,
                "baseline_method": "aligned_logistic_l2",
                "aligned_mean": pivot[metric]["aligned_logistic_l2"].mean(),
                "selected_mean": pivot[metric]["mucff"].mean(),
                "mean_gain": difference.mean(),
                **paired,
            }
        )
    pd.DataFrame(rows).to_csv(
        args.output / "selected_framework_summary.csv", index=False
    )

    arrays: dict[str, np.ndarray] = {}
    for task_directory in sorted(path for path in args.run.iterdir() if path.is_dir()):
        prediction_path = task_directory / "predictions.npz"
        if not prediction_path.is_file():
            continue
        with np.load(prediction_path, allow_pickle=False) as stored:
            for label in ("y_oof", "y_eval"):
                arrays[f"{task_directory.name}__{label}"] = stored[label]
            for method in METHODS:
                for partition in ("oof", "eval"):
                    key = f"{method}_{partition}"
                    arrays[f"{task_directory.name}__{key}"] = stored[key]
    if not arrays:
        raise ValueError(f"No prediction archives found under {args.run}")
    np.savez_compressed(args.output / "reference_predictions.npz", **arrays)


if __name__ == "__main__":
    main()
