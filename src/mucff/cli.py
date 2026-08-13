"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .extended_baselines import run_extended_baselines
from .experiment import compare_methods, load_config, run_task, summarize, write_task_results
from .ledger import discover_ledgers, load_ledger
from .metrics import binary_metrics, select_mcc_threshold
from .reporting import make_result_figures
from .verification import verify_predictions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mucff")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run MuCFF and matched controls")
    run.add_argument("--data", type=Path, default=Path("data/processed"))
    run.add_argument("--config", type=Path, default=Path("configs/main_experiment.json"))
    run.add_argument("--output", type=Path, default=Path("outputs/main_experiment"))
    run.add_argument("--tasks", nargs="*", default=None)
    run.add_argument("--resume", action="store_true")
    verify = subparsers.add_parser("verify", help="Verify predictions against reference arrays")
    verify.add_argument("--output", type=Path, default=Path("outputs/main_experiment"))
    verify.add_argument(
        "--reference",
        type=Path,
        default=Path("results/reference/reference_predictions.npz"),
    )
    verify.add_argument("--tolerance", type=float, default=2e-7)
    verify.add_argument("--tasks", nargs="*", default=None)
    benchmark = subparsers.add_parser("benchmark", help="Run extended learned-fusion baselines")
    benchmark.add_argument("--data", type=Path, default=Path("data/processed"))
    benchmark.add_argument("--config", type=Path, default=Path("configs/main_experiment.json"))
    benchmark.add_argument("--output", type=Path, default=Path("outputs/extended_baselines"))
    benchmark.add_argument("--tasks", nargs="*", default=None)
    benchmark.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    plot = subparsers.add_parser("plot", help="Generate quantitative result figures")
    plot.add_argument("--reference", type=Path, default=Path("results/reference"))
    plot.add_argument("--output", type=Path, default=Path("outputs/figures"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify":
        task_ids = set(args.tasks) if args.tasks else None
        report = verify_predictions(args.output, args.reference, args.tolerance, task_ids)
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
    if args.command == "benchmark":
        rows = []
        args.output.mkdir(parents=True, exist_ok=True)
        for path in discover_ledgers(args.data):
            ledger = load_ledger(path)
            if selected is not None and ledger.task_id not in selected:
                continue
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
                row = {"task_id": ledger.task_id, "method": method}
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
        metrics = pd.DataFrame(rows)
        metrics.to_csv(args.output / "task_metrics.csv", index=False)
        summarize(metrics).to_csv(args.output / "method_summary.csv", index=False)
        return
    all_metrics = []
    for path in discover_ledgers(args.data):
        ledger = load_ledger(path)
        if selected is not None and ledger.task_id not in selected:
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
    combined = pd.concat(all_metrics, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output / "task_metrics.csv", index=False)
    summarize(combined).to_csv(args.output / "method_summary.csv", index=False)
    compare_methods(combined).to_csv(args.output / "auc_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
