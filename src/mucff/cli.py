"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .extended_baselines import run_attention_benchmark, run_extended_baselines
from .experiment import (
    aggregate_evaluation_tasks,
    compare_methods,
    load_config,
    run_task,
    summarize,
    write_task_results,
)
from .ledger import discover_ledgers, load_ledger
from .metrics import binary_metrics, select_mcc_threshold
from .reporting import make_result_figures
from .robustness import run_robustness, summarize_robustness
from .verification import METHODS, verify_predictions


AUXILIARY_TASKS = frozenset({"ienhancer_recognition", "ienhancer_strength"})
PRIMARY_TASKS = frozenset(
    {
        "gue_emp_H3",
        "gue_human_tf_0",
        "gue_mouse_0",
        "gue_prom_300_all",
        "gue_prom_core_all",
        "ipromoter_hs_notata",
        "ipromoter_hs_tata",
        "ipromoter_mm_notata",
        "ipromoter_mm_tata",
        "snnrice6ma_rice_chen",
    }
)


def _matches_task(task_id: str, requested: set[str] | None) -> bool:
    if requested is None:
        return True
    return task_id in requested or any(
        task_id.startswith(f"{name}__fold") for name in requested
    )


def _include_task(
    task_id: str,
    requested: set[str] | None,
    include_auxiliary: bool,
) -> bool:
    if requested is not None:
        return _matches_task(task_id, requested)
    return include_auxiliary or task_id not in AUXILIARY_TASKS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mucff")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run MuCFF and matched controls")
    run.add_argument("--data", type=Path, default=Path("data/ledgers"))
    run.add_argument("--config", type=Path, default=Path("configs/main_experiment.json"))
    run.add_argument("--output", type=Path, default=Path("outputs/main_experiment"))
    run.add_argument("--tasks", nargs="*", default=None)
    run.add_argument("--include-auxiliary", action="store_true")
    run.add_argument("--resume", action="store_true")
    verify = subparsers.add_parser("verify", help="Verify predictions against reference arrays")
    verify.add_argument("--output", type=Path, default=Path("outputs/main_experiment"))
    verify.add_argument(
        "--reference",
        type=Path,
        default=Path("results/reference/reference_predictions.npz"),
    )
    verify.add_argument("--tolerance", type=float, default=3e-2)
    verify.add_argument("--metric-tolerance", type=float, default=5e-4)
    verify.add_argument("--tasks", nargs="*", default=None)
    verify.add_argument("--methods", nargs="*", default=None)
    benchmark = subparsers.add_parser("benchmark", help="Run extended learned-fusion baselines")
    benchmark.add_argument("--data", type=Path, default=Path("data/ledgers"))
    benchmark.add_argument("--config", type=Path, default=Path("configs/main_experiment.json"))
    benchmark.add_argument("--output", type=Path, default=Path("outputs/extended_baselines"))
    benchmark.add_argument("--tasks", nargs="*", default=None)
    benchmark.add_argument("--include-auxiliary", action="store_true")
    benchmark.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    benchmark.add_argument(
        "--suite",
        choices=["all", "attention"],
        default="all",
        help="Run all extended baselines or only the matched QKV attention suite",
    )
    plot = subparsers.add_parser("plot", help="Generate quantitative result figures")
    plot.add_argument("--reference", type=Path, default=Path("results/reference"))
    plot.add_argument("--output", type=Path, default=Path("outputs/figures"))
    robustness = subparsers.add_parser(
        "robustness", help="Evaluate source ablation and deployment perturbations"
    )
    robustness.add_argument("--data", type=Path, default=Path("data/ledgers"))
    robustness.add_argument("--config", type=Path, default=Path("configs/main_experiment.json"))
    robustness.add_argument("--output", type=Path, default=Path("outputs/robustness"))
    robustness.add_argument("--tasks", nargs="*", default=None)
    robustness.add_argument("--include-auxiliary", action="store_true")
    robustness.add_argument("--repeats", type=int, default=20)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify":
        task_ids = set(args.tasks) if args.tasks else set(PRIMARY_TASKS)
        methods = tuple(args.methods) if args.methods else METHODS
        report = verify_predictions(
            args.output,
            args.reference,
            args.tolerance,
            args.metric_tolerance,
            task_ids,
            methods,
        )
        report.to_csv(args.output / "verification_report.csv", index=False)
        print(f"verified {report.task_id.nunique()} tasks", flush=True)
        return
    if args.command == "plot":
        make_result_figures(args.reference, args.output)
        print(f"wrote figures to {args.output}", flush=True)
        return
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = load_config(args.config)
    selected = set(args.tasks) if args.tasks else None
    if args.command == "robustness":
        rows = []
        args.output.mkdir(parents=True, exist_ok=True)
        for path in discover_ledgers(args.data):
            ledger = load_ledger(path)
            if not _include_task(
                ledger.task_id, selected, args.include_auxiliary
            ):
                continue
            result = run_robustness(ledger, config, settings, args.repeats)
            task_output = args.output / ledger.task_id
            task_output.mkdir(parents=True, exist_ok=True)
            result.to_csv(task_output / "robustness_metrics.csv", index=False)
            rows.append(result)
            print(f"completed {ledger.task_id}", flush=True)
        if not rows:
            raise ValueError("No requested tasks were found.")
        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv(args.output / "robustness_task_metrics.csv", index=False)
        summarize_robustness(combined).to_csv(
            args.output / "robustness_summary.csv", index=False
        )
        return
    if args.command == "benchmark":
        rows = []
        args.output.mkdir(parents=True, exist_ok=True)
        for path in discover_ledgers(args.data):
            ledger = load_ledger(path)
            if not _include_task(
                ledger.task_id, selected, args.include_auxiliary
            ):
                continue
            if args.suite == "attention":
                predictions = run_attention_benchmark(ledger, config, args.device)
            else:
                predictions = run_extended_baselines(ledger, config, args.device)
            arrays = {"y_oof": ledger.y_oof, "y_eval": ledger.y_eval}
            for method, (oof_probability, eval_probability) in predictions.items():
                threshold = select_mcc_threshold(
                    ledger.y_oof,
                    oof_probability,
                    float(settings["threshold_quantile_low"]),
                    float(settings["threshold_quantile_high"]),
                    int(settings["threshold_quantile_count"]),
                )
                row = {
                    "task_id": ledger.task_id,
                    "method": method,
                    "n_sources": ledger.n_sources,
                }
                row.update(binary_metrics(ledger.y_eval, eval_probability, threshold))
                rows.append(row)
                arrays[f"{method}_oof"] = oof_probability.astype("float32")
                arrays[f"{method}_eval"] = eval_probability.astype("float32")
            task_output = args.output / ledger.task_id
            task_output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row for row in rows if row["task_id"] == ledger.task_id]).to_csv(
                task_output / "metrics.csv", index=False
            )
            import numpy as np

            np.savez_compressed(task_output / "predictions.npz", **arrays)
            print(f"completed {ledger.task_id}", flush=True)
        if not rows:
            raise ValueError("No requested tasks were found.")
        metrics = aggregate_evaluation_tasks(args.output, pd.DataFrame(rows), settings)
        metrics.to_csv(args.output / "task_metrics.csv", index=False)
        summarize(metrics).to_csv(args.output / "method_summary.csv", index=False)
        return
    all_metrics = []
    for path in discover_ledgers(args.data):
        ledger = load_ledger(path)
        if not _include_task(ledger.task_id, selected, args.include_auxiliary):
            continue
        task_metrics = args.output / ledger.task_id / "metrics.csv"
        task_predictions = args.output / ledger.task_id / "predictions.npz"
        if args.resume and task_metrics.is_file() and task_predictions.is_file():
            all_metrics.append(pd.read_csv(task_metrics))
            print(f"retained {ledger.task_id}", flush=True)
            continue
        metrics, arrays = run_task(ledger, config, settings)
        write_task_results(args.output, ledger, metrics, arrays)
        all_metrics.append(metrics)
        print(f"completed {ledger.task_id}", flush=True)
    if not all_metrics:
        raise ValueError("No requested tasks were found.")
    combined = aggregate_evaluation_tasks(
        args.output, pd.concat(all_metrics, ignore_index=True), settings
    )
    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output / "task_metrics.csv", index=False)
    summarize(combined).to_csv(args.output / "method_summary.csv", index=False)
    compare_methods(combined).to_csv(args.output / "auc_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
