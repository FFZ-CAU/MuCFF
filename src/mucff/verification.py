"""Numerical verification against released reference predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("mucff", "sparse_aligned_control")


def verify_predictions(
    output_root: str | Path,
    reference_archive: str | Path,
    tolerance: float = 2e-7,
    task_ids: set[str] | None = None,
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
            missing = sorted(task_ids.difference(available_task_ids))
            if missing:
                raise ValueError(f"Requested tasks are absent from the reference: {missing}")
            available_task_ids = [task_id for task_id in available_task_ids if task_id in task_ids]
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
                for method in METHODS:
                    for partition in ("oof", "eval"):
                        observed_values = observed[f"{method}_{partition}"]
                        reference_values = reference[f"{task_id}__{method}_{partition}"]
                        difference = np.abs(observed_values - reference_values)
                        maximum = float(difference.max(initial=0.0))
                        rows.append(
                            {
                                "task_id": task_id,
                                "method": method,
                                "partition": partition,
                                "maximum_absolute_difference": maximum,
                                "verified": maximum <= tolerance,
                            }
                        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No requested tasks were found.")
    if not result.verified.all():
        failed = result.loc[~result.verified, ["task_id", "method", "partition"]]
        raise AssertionError(f"Prediction verification failed:\n{failed.to_string(index=False)}")
    return result
