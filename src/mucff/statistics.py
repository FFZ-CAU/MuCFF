"""Across-task summaries for matched fusion comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


EPSILON = 1e-12


def _paired_summary(differences: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=float)
    generator = np.random.default_rng(20260810)
    bootstrap = values[
        generator.integers(0, len(values), size=(200_000, len(values)))
    ].mean(axis=1)
    nonzero = values[np.abs(values) > EPSILON]
    p_value = (
        float(wilcoxon(nonzero, alternative="greater", method="exact").pvalue)
        if nonzero.size
        else 1.0
    )
    return {
        "mean_auc_difference": float(values.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "wins": int(np.sum(values > EPSILON)),
        "ties": int(np.sum(np.abs(values) <= EPSILON)),
        "losses": int(np.sum(values < -EPSILON)),
        "wilcoxon_one_sided_p": p_value,
    }


def compare_auc(metrics: pd.DataFrame, reference_method: str) -> pd.DataFrame:
    required = {"task_id", "method", "auc"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metric table is missing columns: {sorted(missing)}")
    table = metrics.pivot(index="task_id", columns="method", values="auc")
    if reference_method not in table:
        raise ValueError(f"Reference method is absent: {reference_method}")
    rows = []
    for method in sorted(name for name in table.columns if name != reference_method):
        pair = table[[reference_method, method]].dropna()
        row: dict[str, object] = {
            "reference_method": reference_method,
            "comparison_method": method,
            "task_count": int(len(pair)),
        }
        row.update(_paired_summary((pair[reference_method] - pair[method]).to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_auc_difference", ascending=False)
