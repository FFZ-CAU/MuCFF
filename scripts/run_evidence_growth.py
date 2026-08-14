#!/usr/bin/env python
"""Evaluate the ordered evidence-bank expansion used in the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mucff.fusion import crossfit_aligned_control, crossfit_mucff
from mucff.ledger import discover_ledgers, load_ledger
from mucff.experiment import load_config
from mucff.metrics import binary_metrics, select_mcc_threshold


STAGES = (
    ("foundation_local_bank", frozenset()),
    ("plus_rc_motif_experts", frozenset({"E03", "E04", "E05"})),
    ("plus_biophysical_grammar", frozenset({"E02", "E03", "E04", "E05"})),
    ("plus_positional_biophysical_grammar", frozenset({"E01", "E02", "E03", "E04", "E05"})),
)


def threshold(labels, probabilities, settings):
    return select_mcc_threshold(
        labels,
        probabilities,
        float(settings["threshold_quantile_low"]),
        float(settings["threshold_quantile_high"]),
        int(settings["threshold_quantile_count"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/main_experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evidence_growth"))
    args = parser.parse_args()

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = load_config(args.config)
    rows = []
    for ledger_path in discover_ledgers(args.data):
        ledger = load_ledger(ledger_path)
        non_expert = [index for index, source_id in enumerate(ledger.source_ids) if not source_id.startswith("E")]
        for stage_name, expert_ids in STAGES:
            retained = non_expert + [
                index for index, source_id in enumerate(ledger.source_ids) if source_id in expert_ids
            ]
            prediction = crossfit_aligned_control(
                ledger.oof_scores[:, retained],
                ledger.y_oof,
                ledger.eval_scores[:, retained],
                ledger.task_id,
                config,
            )
            row = {
                "task_id": ledger.task_id,
                "stage": stage_name,
                "n_sources": len(retained),
            }
            row.update(
                binary_metrics(
                    ledger.y_eval,
                    prediction.eval_probability,
                    threshold(ledger.y_oof, prediction.oof_probability, settings),
                )
            )
            rows.append(row)

        prediction = crossfit_mucff(
            ledger.oof_scores,
            ledger.y_oof,
            ledger.eval_scores,
            ledger.task_id,
            ledger.source_families,
            config,
        )
        row = {"task_id": ledger.task_id, "stage": "complete_mucff", "n_sources": ledger.n_sources}
        row.update(
            binary_metrics(
                ledger.y_eval,
                prediction.eval_probability,
                threshold(ledger.y_oof, prediction.oof_probability, settings),
            )
        )
        rows.append(row)
        print(f"completed {ledger.task_id}", flush=True)

    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "evidence_growth_task_metrics.csv", index=False)
    summary = frame.groupby("stage", as_index=False).agg(
        nominal_sources=("n_sources", lambda values: f"{max(values)}/{min(values)}"),
        mean_auc=("auc", "mean"),
        mean_auprc=("auprc", "mean"),
        mean_mcc=("mcc", "mean"),
        mean_accuracy=("accuracy", "mean"),
    )
    order = {name: index for index, (name, _) in enumerate(STAGES)} | {"complete_mucff": len(STAGES)}
    summary["order"] = summary.stage.map(order)
    summary = summary.sort_values("order").drop(columns="order")
    base_auc = float(summary.iloc[0].mean_auc)
    summary["cumulative_auc_gain"] = summary.mean_auc - base_auc
    summary["incremental_auc_gain"] = summary.mean_auc.diff()
    summary.to_csv(args.output / "evidence_growth_summary.csv", index=False)


if __name__ == "__main__":
    main()
