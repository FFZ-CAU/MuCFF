#!/usr/bin/env python
"""Evaluate the ordered construction of the MuCFF evidence bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mucff.experiment import aggregate_evaluation_tasks, load_config
from mucff.fusion import (
    crossfit_aligned_control,
    crossfit_mucff,
    crossfit_routed_l2,
)
from mucff.ledger import discover_ledgers, load_ledger
from mucff.metrics import binary_metrics, select_mcc_threshold
from mucff.statistics import _paired_summary


GRAMMAR_ORDER = (
    "rc_multiscale_seed0",
    "rc_multiscale_seed1",
    "rc_multiscale_seed2",
    "rc_multiscale_biophysical_seed3",
    "rc_multiscale_biophysical_position_seed4",
)
STAGE_ORDER = (
    "base_evidence",
    "foundation_interactions",
    "local_foundation_fusion",
    "rc_grammar_1",
    "rc_grammar_2",
    "rc_grammar_3",
    "biophysical_grammar",
    "position_aware_grammar",
    "reliability_routed_l2",
    "mucff_dual_decision",
)


def _threshold(labels, probabilities, settings):
    return select_mcc_threshold(
        labels,
        probabilities,
        float(settings["threshold_quantile_low"]),
        float(settings["threshold_quantile_high"]),
        int(settings["threshold_quantile_count"]),
    )


def _stage_indices(ledger, source_tiers):
    names = np.asarray(ledger.source_ids)
    tiers = np.asarray([source_tiers[name] for name in names])
    selected = list(np.flatnonzero(tiers == "Primitive"))
    yield STAGE_ORDER[0], selected.copy()
    selected.extend(np.flatnonzero(tiers == "Foundation interaction").tolist())
    yield STAGE_ORDER[1], sorted(selected)
    selected.extend(np.flatnonzero(tiers == "Local or derived").tolist())
    yield STAGE_ORDER[2], sorted(selected)
    for offset, source_name in enumerate(GRAMMAR_ORDER, start=3):
        matches = np.flatnonzero(names == source_name)
        if matches.size:
            selected.append(int(matches[0]))
        yield STAGE_ORDER[offset], sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ledgers"))
    parser.add_argument("--metadata", type=Path, default=Path("data/source_metadata.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/main_experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evidence_growth"))
    args = parser.parse_args()

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = load_config(args.config)
    metadata = pd.read_csv(args.metadata).set_index("source_name")
    source_tiers = metadata.source_tier.to_dict()
    rows = []
    run_root = args.output / "runs"

    for ledger_path in discover_ledgers(args.data):
        ledger = load_ledger(ledger_path)
        if ledger.task_id.startswith("ienhancer_"):
            continue
        arrays = {"y_oof": ledger.y_oof, "y_eval": ledger.y_eval}
        for stage, indices in _stage_indices(ledger, source_tiers):
            prediction = crossfit_aligned_control(
                ledger.oof_scores[:, indices],
                ledger.y_oof,
                ledger.eval_scores[:, indices],
                ledger.task_id,
                config,
            )
            threshold = _threshold(ledger.y_oof, prediction.oof_probability, settings)
            row = {"task_id": ledger.task_id, "method": stage, "n_sources": len(indices)}
            row.update(binary_metrics(ledger.y_eval, prediction.eval_probability, threshold))
            rows.append(row)
            arrays[f"{stage}_oof"] = prediction.oof_probability
            arrays[f"{stage}_eval"] = prediction.eval_probability

        routed = crossfit_routed_l2(
            ledger.oof_scores,
            ledger.y_oof,
            ledger.eval_scores,
            ledger.task_id,
            ledger.source_families,
            config,
        )
        final = crossfit_mucff(
            ledger.oof_scores,
            ledger.y_oof,
            ledger.eval_scores,
            ledger.task_id,
            ledger.source_families,
            config,
        )
        for stage, prediction in (
            (STAGE_ORDER[-2], routed),
            (STAGE_ORDER[-1], final),
        ):
            threshold = _threshold(ledger.y_oof, prediction.oof_probability, settings)
            row = {"task_id": ledger.task_id, "method": stage, "n_sources": ledger.n_sources}
            row.update(binary_metrics(ledger.y_eval, prediction.eval_probability, threshold))
            rows.append(row)
            arrays[f"{stage}_oof"] = prediction.oof_probability
            arrays[f"{stage}_eval"] = prediction.eval_probability

        task_output = run_root / ledger.task_id
        task_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(task_output / "predictions.npz", **arrays)
        print(f"completed {ledger.task_id}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    run_metrics = pd.DataFrame(rows)
    run_metrics.to_csv(args.output / "evidence_growth_run_metrics.csv", index=False)
    task_metrics = aggregate_evaluation_tasks(run_root, run_metrics, settings)
    task_metrics.to_csv(args.output / "evidence_growth_task_metrics.csv", index=False)

    table = task_metrics.pivot(index="task_id", columns="method", values="auc")
    summary_rows = []
    previous = None
    for order, stage in enumerate(STAGE_ORDER):
        selected = task_metrics.loc[task_metrics.method.eq(stage)]
        row = {
            "order": order,
            "stage": stage,
            "nominal_source_count": int(selected.n_sources.max()),
            "minimum_source_count": int(selected.n_sources.min()),
            "mean_auc": selected.auc.mean(),
            "mean_auprc": selected.auprc.mean(),
            "mean_mcc": selected.mcc.mean(),
            "mean_accuracy": selected.accuracy.mean(),
        }
        if previous is not None:
            paired = _paired_summary((table[stage] - table[previous]).to_numpy())
            row.update({f"step_{key}": value for key, value in paired.items()})
        summary_rows.append(row)
        previous = stage
    pd.DataFrame(summary_rows).to_csv(
        args.output / "evidence_growth_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
