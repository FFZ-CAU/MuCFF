import json
import platform
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mucff.experiment import load_config, run_task
from mucff.ledger import load_ledger
from mucff.verification import verify_predictions


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "snnrice6ma_rice_chen__fold00"


@pytest.fixture(scope="module")
def rice_fold_result():
    settings_path = ROOT / "configs" / "main_experiment.json"
    config = load_config(settings_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    ledger = load_ledger(
        ROOT / "data" / "ledgers" / TASK_ID / "expanded_ledger.npz"
    )
    metrics, arrays = run_task(ledger, config, settings)
    return ledger, metrics, arrays


def test_rice_fold_reference_metrics(rice_fold_result):
    _, metrics, _ = rice_fold_result
    indexed = metrics.set_index("method")
    expected = {
        "aligned_logistic_l2": 0.9278150826446281,
        "routed_logistic_l2": 0.9258780991735537,
    }
    for method, auc in expected.items():
        assert indexed.loc[method, "auc"] == pytest.approx(auc, abs=1e-12)

    reference_auc = 0.9262654958677685
    tolerance = 1e-12 if platform.system() == "Linux" else 3e-3
    assert indexed.loc["mucff", "auc"] == pytest.approx(
        reference_auc, abs=tolerance
    )


def test_rice_fold_reference_predictions(tmp_path, rice_fold_result):
    ledger, _, arrays = rice_fold_result
    task_directory = tmp_path / ledger.task_id
    task_directory.mkdir()
    np.savez_compressed(task_directory / "predictions.npz", **arrays)
    report = verify_predictions(
        tmp_path,
        ROOT / "results" / "reference" / "reference_predictions.npz",
        task_ids={ledger.task_id},
        methods=("aligned_logistic_l2", "routed_logistic_l2"),
    )
    assert report.task_id.unique().tolist() == [ledger.task_id]
    assert report.verified.all()


def test_grouped_reference_selector_expands_rice_folds():
    with np.load(
        ROOT / "results" / "reference" / "reference_predictions.npz",
        allow_pickle=False,
    ) as stored:
        rice_tasks = {
            key.removesuffix("__y_eval")
            for key in stored.files
            if key.startswith("snnrice6ma_rice_chen__fold")
            and key.endswith("__y_eval")
        }
    assert len(rice_tasks) == 10


def test_reference_archive_contains_expected_labels():
    with np.load(
        ROOT / "results" / "reference" / "reference_predictions.npz",
        allow_pickle=False,
    ) as stored:
        labels = stored[f"{TASK_ID}__y_eval"]
        probability = stored[f"{TASK_ID}__mucff_eval"]
    assert labels.shape == probability.shape
    assert roc_auc_score(labels, probability) == pytest.approx(
        0.9262654958677685, abs=1e-12
    )
