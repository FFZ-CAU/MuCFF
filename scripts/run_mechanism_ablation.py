#!/usr/bin/env python
"""Run the matched residual-state-by-decision ablation on the common ledger."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mucff.experiment import load_config
from mucff.fusion import MuCFFConfig, make_sparse_decision, predict_probability, stable_seed
from mucff.ledger import discover_ledgers, load_ledger
from mucff.metrics import binary_metrics, select_mcc_threshold
from mucff.representation import aligned_state, mucff_state, select_anchor


def l2_decision(config: MuCFFConfig):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.regularization_c,
            solver="lbfgs",
            max_iter=config.max_iterations,
            class_weight="balanced",
            random_state=config.model_seed,
        ),
    )


def crossfit_state(
    oof_scores,
    labels,
    eval_scores,
    task_id,
    config,
    include_residual,
    sparse,
):
    splitter = StratifiedKFold(
        config.outer_folds,
        shuffle=True,
        random_state=stable_seed(config.seed_base, task_id, "fusion_benchmark_common_cv"),
    )
    estimator = make_sparse_decision(config) if sparse else l2_decision(config)
    oof_probability = np.zeros(labels.size, dtype=np.float32)
    eval_probabilities = []
    for train_index, validation_index in splitter.split(oof_scores, labels):
        if include_residual:
            anchor_index = select_anchor(
                oof_scores[train_index], labels[train_index],
                config.probability_epsilon,
            )
            train_state = mucff_state(
                oof_scores[train_index], anchor_index, config.probability_epsilon,
            )
            validation_state = mucff_state(
                oof_scores[validation_index], anchor_index, config.probability_epsilon,
            )
            eval_state = mucff_state(
                eval_scores, anchor_index, config.probability_epsilon,
            )
        else:
            train_state = aligned_state(oof_scores[train_index], config.probability_epsilon)
            validation_state = aligned_state(oof_scores[validation_index], config.probability_epsilon)
            eval_state = aligned_state(eval_scores, config.probability_epsilon)
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_state, labels[train_index])
        oof_probability[validation_index] = predict_probability(
            model, validation_state, config.probability_epsilon
        )
        eval_probabilities.append(predict_probability(model, eval_state, config.probability_epsilon))
    return oof_probability, np.mean(eval_probabilities, axis=0)


def paired_summary(frame, comparison_method, reference_method, metric):
    table = frame.pivot(index="task_id", columns="method", values=metric)
    differences = (table[comparison_method] - table[reference_method]).to_numpy()
    generator = np.random.default_rng(20260810)
    bootstrap = differences[
        generator.integers(0, len(differences), size=(200_000, len(differences)))
    ].mean(axis=1)
    nonzero = differences[np.abs(differences) > 1e-12]
    p_value = float(wilcoxon(nonzero, alternative="greater", method="exact").pvalue)
    return {
        "comparison": f"{comparison_method}_vs_{reference_method}",
        "metric": metric,
        "comparison_method_mean": float(table[comparison_method].mean()),
        "reference_method_mean": float(table[reference_method].mean()),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "wins": int(np.sum(differences > 1e-12)),
        "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "losses": int(np.sum(differences < -1e-12)),
        "wilcoxon_one_sided_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/main_experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mechanism_ablation"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = load_config(args.config)
    methods = (
        ("aligned_l2", False, False),
        ("residual_l2", True, False),
        ("aligned_sparse", False, True),
        ("mucff", True, True),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    task_metrics_path = args.output / "mechanism_ablation_task_metrics.csv"
    rows = []
    if args.resume and task_metrics_path.is_file():
        rows = pd.read_csv(task_metrics_path).to_dict("records")
    completed = {
        task_id
        for task_id, group in pd.DataFrame(rows).groupby("task_id")
        if set(group["method"]) == {method[0] for method in methods}
    } if rows else set()
    for path in discover_ledgers(args.data):
        ledger = load_ledger(path)
        if ledger.task_id in completed:
            print(f"retained {ledger.task_id}", flush=True)
            continue
        for method, include_residual, sparse in methods:
            oof, evaluation = crossfit_state(
                ledger.oof_scores, ledger.y_oof, ledger.eval_scores, ledger.task_id,
                config, include_residual, sparse,
            )
            threshold = select_mcc_threshold(
                ledger.y_oof, oof,
                float(settings["threshold_quantile_low"]),
                float(settings["threshold_quantile_high"]),
                int(settings["threshold_quantile_count"]),
            )
            row = {"task_id": ledger.task_id, "method": method, "oof_auc": roc_auc_score(ledger.y_oof, oof)}
            row.update(binary_metrics(ledger.y_eval, evaluation, threshold))
            rows.append(row)
        pd.DataFrame(rows).to_csv(task_metrics_path, index=False)
        print(f"completed {ledger.task_id}", flush=True)

    frame = pd.DataFrame(rows)
    counts = frame.groupby("task_id")["method"].nunique()
    if counts.empty or counts.nunique() != 1 or int(counts.iloc[0]) != len(methods):
        raise RuntimeError(f"Incomplete mechanism-ablation grid: {counts.to_dict()}")
    frame.to_csv(task_metrics_path, index=False)
    frame.groupby("method", as_index=False).mean(numeric_only=True).to_csv(
        args.output / "mechanism_ablation_summary.csv", index=False
    )
    comparisons = []
    for comparison_method, reference_method in (
        ("residual_l2", "aligned_l2"),
        ("mucff", "aligned_sparse"),
        ("mucff", "residual_l2"),
    ):
        for metric in ("auc", "auprc", "mcc", "accuracy", "balanced_accuracy"):
            comparisons.append(
                paired_summary(frame, comparison_method, reference_method, metric)
            )
    pd.DataFrame(comparisons).to_csv(
        args.output / "mechanism_ablation_comparisons.csv", index=False
    )


if __name__ == "__main__":
    main()
