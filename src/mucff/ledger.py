"""Evidence-ledger input and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EvidenceLedger:
    task_id: str
    y_oof: np.ndarray
    y_eval: np.ndarray
    oof_scores: np.ndarray
    eval_scores: np.ndarray
    source_ids: tuple[str, ...]
    source_families: tuple[str, ...]

    @property
    def n_sources(self) -> int:
        return self.oof_scores.shape[1]


def _text_tuple(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in values.tolist())


def validate_ledger(ledger: EvidenceLedger) -> None:
    if ledger.y_oof.ndim != 1 or ledger.y_eval.ndim != 1:
        raise ValueError("Labels must be one-dimensional arrays.")
    if ledger.oof_scores.ndim != 2 or ledger.eval_scores.ndim != 2:
        raise ValueError("Score matrices must be two-dimensional arrays.")
    if ledger.oof_scores.shape[0] != ledger.y_oof.size:
        raise ValueError("OOF labels and scores have different sample counts.")
    if ledger.eval_scores.shape[0] != ledger.y_eval.size:
        raise ValueError("Evaluation labels and scores have different sample counts.")
    if ledger.oof_scores.shape[1] != ledger.eval_scores.shape[1]:
        raise ValueError("OOF and evaluation ledgers have different source counts.")
    if len(ledger.source_ids) != ledger.n_sources:
        raise ValueError("Source identifiers do not match the score columns.")
    if len(ledger.source_families) != ledger.n_sources:
        raise ValueError("Source families do not match the score columns.")
    if len(set(ledger.source_ids)) != ledger.n_sources:
        raise ValueError("Source identifiers must be unique within a task.")
    if set(np.unique(ledger.y_oof)) != {0, 1}:
        raise ValueError("OOF labels must contain both binary classes.")
    if set(np.unique(ledger.y_eval)) != {0, 1}:
        raise ValueError("Evaluation labels must contain both binary classes.")
    if not np.isfinite(ledger.oof_scores).all() or not np.isfinite(ledger.eval_scores).all():
        raise ValueError("Score matrices contain non-finite values.")


def load_ledger(path: str | Path) -> EvidenceLedger:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Evidence ledger not found: {source}")
    with np.load(source, allow_pickle=False) as stored:
        required = {"y_oof", "oof_scores", "source_families"}
        missing = required.difference(stored.files)
        if missing:
            raise ValueError(f"Evidence ledger is missing arrays: {sorted(missing)}")
        eval_label_key = "y_eval" if "y_eval" in stored.files else "y_test"
        eval_score_key = "eval_scores" if "eval_scores" in stored.files else "test_scores"
        source_key = "source_ids" if "source_ids" in stored.files else "source_names"
        secondary = {eval_label_key, eval_score_key, source_key}
        if any(key not in stored.files for key in secondary):
            raise ValueError("Evidence ledger is missing evaluation or source arrays.")
        ledger = EvidenceLedger(
            task_id=(
                str(stored["task_id"].item())
                if "task_id" in stored.files
                else source.parent.name
            ),
            y_oof=stored["y_oof"].astype(np.int64),
            y_eval=stored[eval_label_key].astype(np.int64),
            oof_scores=stored["oof_scores"].astype(np.float64),
            eval_scores=stored[eval_score_key].astype(np.float64),
            source_ids=_text_tuple(stored[source_key]),
            source_families=_text_tuple(stored["source_families"]),
        )
    validate_ledger(ledger)
    return ledger


def discover_ledgers(root: str | Path) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Ledger directory not found: {directory}")
    paths = sorted(directory.glob("*/evidence_ledger.npz"))
    if not paths:
        paths = sorted(directory.glob("*/expanded_ledger.npz"))
    if not paths:
        raise FileNotFoundError(f"No evidence ledgers found under: {directory}")
    return paths
