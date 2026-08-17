"""Numerical verification against released reference predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


METHODS = (
    "mucff",
    "aligned_logistic_l2",
    "routed_logistic_l2",
    "aligned_xgboost",
)


def verify_predictions(
    output_root: str | Path,
    reference_archive: str | Path,
    tolerance: float = 3e-2,
    metric_tolerance: float = 5e-4,
    task_ids: set[str] | None = None,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    output_directory = Path(output_root)
    rows = []
    with np.load(reference_archive, allow_pickle=False) as reference:
        available_task_ids = sorted(
            key.removesuffix("__y_eval")
            for key in reference.files
            if key.endswith("__y_eval")
        )
        if task_ids is not None:
            matched = {
                task_id
                for task_id in available_task_ids
                if task_id in task_ids
                or any(task_id.startswith(f"{name}__fold") for name in task_ids)
            }
            missing = sorted(
                name
                for name in task_ids
                if name not in available_task_ids
                and not any(
                    task_id.startswith(f"{name}__fold")
                    for task_id in available_task_ids
                )
            )
            if missing:
                raise ValueError(f"Requested tasks are absent from the reference: {missing}")
            available_task_ids = [
                task_id for task_id in available_task_ids if task_id in matched
            ]
        for task_id in available_task_ids:
            prediction_path = output_directory / task_id / "predictions.npz"
            if not prediction_path.is_file():
                raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
            with np.load(prediction_path, allow_pickle=False) as observed:
                for label in ("y_oof", "y_eval"):
                    if not np.array_equal(
                        observed[label], reference[f"{task_id}__{label}"]
                    ):
                        raise AssertionError(f"Label mismatch for {task_id}: {label}")
                for method in methods:
                    for partition in ("oof", "eval"):
                        observed_values = observed[f"{method}_{partition}"]
                        reference_values = reference[f"{task_id}__{method}_{partition}"]
                        difference = np.abs(observed_values - reference_values)
                        maximum = float(difference.max(initial=0.0))
                        mean = float(difference.mean())
                        label_key = "y_oof" if partition == "oof" else "y_eval"
                        auc_difference = abs(
                            float(roc_auc_score(observed[label_key], observed_values))
                            - float(
                                roc_auc_score(
                                    reference[f"{task_id}__{label_key}"],
                                    reference_values,
                                )
                            )
                        )
                        rows.append(
                            {
                                "task_id": task_id,
                                "method": method,
                                "partition": partition,
                                "maximum_absolute_difference": maximum,
                                "mean_absolute_difference": mean,
                                "absolute_auc_difference": auc_difference,
                                "verified": (
                                    maximum <= tolerance
                                    and mean <= tolerance / 10.0
                                    and auc_difference <= metric_tolerance
                                ),
                            }
                        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No requested tasks were found.")
    if not result.verified.all():
        failed = result.loc[
            ~result.verified,
            [
                "task_id",
                "method",
                "partition",
                "maximum_absolute_difference",
                "absolute_auc_difference",
            ],
        ]
        raise AssertionError(f"Prediction verification failed:\n{failed.to_string(index=False)}")
    return result
