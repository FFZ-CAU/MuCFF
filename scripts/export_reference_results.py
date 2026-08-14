#!/usr/bin/env python
"""Create the reproducibility archive from a completed MuCFF run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = (
    "auc",
    "auprc",
    "mcc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)


def paired_summary(values: np.ndarray, seed: int = 20260810) -> dict[str, float | int]:
    differences = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(200_000, differences.size))
    bootstrap = differences[indices].mean(axis=1)
    nonzero = differences[np.abs(differences) > 1e-12]
    p_value = (
        float(wilcoxon(nonzero, alternative="greater", method="exact").pvalue)
        if nonzero.size
        else 1.0
    )
    return {
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "wins": int(np.sum(differences > 1e-12)),
        "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "losses": int(np.sum(differences < -1e-12)),
        "wilcoxon_one_sided_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("outputs/main_experiment"))
    parser.add_argument("--output", type=Path, default=Path("results/reference"))
    args = parser.parse_args()

    metrics = pd.read_csv(args.run / "task_metrics.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "reference_task_metrics.csv", index=False)

    summary = metrics.groupby("method", as_index=False)[list(METRICS)].mean()
    summary.to_csv(args.output / "reference_method_summary.csv", index=False)

    pivoted = {
        metric: metrics.pivot(index="task_id", columns="method", values=metric)
        for metric in METRICS
    }
    comparison_rows = []
    for metric, table in pivoted.items():
        differences = (table["mucff"] - table["sparse_aligned_control"]).to_numpy()
        row: dict[str, float | int | str] = {
            "metric": metric,
            "sparse_aligned_control_mean": float(table["sparse_aligned_control"].mean()),
            "mucff_mean": float(table["mucff"].mean()),
            "mucff_minus_control": float(differences.mean()),
        }
        row.update(paired_summary(differences))
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(
        args.output / "reference_metric_summary.csv", index=False
    )

    arrays: dict[str, np.ndarray] = {}
    for task_directory in sorted(path for path in args.run.iterdir() if path.is_dir()):
        prediction_path = task_directory / "predictions.npz"
        if not prediction_path.is_file():
            continue
        with np.load(prediction_path, allow_pickle=False) as stored:
            for key in stored.files:
                arrays[f"{task_directory.name}__{key}"] = stored[key]
    if not arrays:
        raise ValueError(f"No task predictions found under {args.run}")
    np.savez_compressed(args.output / "reference_predictions.npz", **arrays)
    print(f"exported {metrics.task_id.nunique()} tasks to {args.output}", flush=True)


if __name__ == "__main__":
    main()
