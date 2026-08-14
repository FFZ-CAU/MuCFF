from pathlib import Path

import pytest

from mucff.experiment import load_config, run_task
from mucff.ledger import load_ledger
from mucff.verification import verify_predictions


ROOT = Path(__file__).resolve().parents[1]


def test_rice_6ma_reference_metrics():
    settings_path = ROOT / "configs" / "main_experiment.json"
    config = load_config(settings_path)
    import json

    threshold_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    ledger = load_ledger(
        ROOT / "data" / "processed" / "snnrice6ma_rice_chen" / "evidence_ledger.npz"
    )
    metrics, _ = run_task(ledger, config, threshold_settings)
    indexed = metrics.set_index("method")
    expected = {
        "mucff": (0.8846548188653452, 0.8927166645567698, 0.5398991206424848),
        "sparse_aligned_control": (
            0.8845408976987924,
            0.89261125287288,
            0.5398991206424848,
        ),
    }
    for method, values in expected.items():
        assert indexed.loc[method, "auc"] == pytest.approx(values[0], abs=1e-12)
        assert indexed.loc[method, "auprc"] == pytest.approx(values[1], abs=1e-12)
        assert indexed.loc[method, "mcc"] == pytest.approx(values[2], abs=1e-12)


def test_rice_6ma_reference_predictions(tmp_path):
    settings_path = ROOT / "configs" / "main_experiment.json"
    config = load_config(settings_path)
    import json
    import numpy as np

    threshold_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    ledger = load_ledger(
        ROOT / "data" / "processed" / "snnrice6ma_rice_chen" / "evidence_ledger.npz"
    )
    _, arrays = run_task(ledger, config, threshold_settings)
    task_directory = tmp_path / ledger.task_id
    task_directory.mkdir()
    np.savez_compressed(task_directory / "predictions.npz", **arrays)
    source_reference = ROOT / "results" / "reference" / "reference_predictions.npz"
    task_reference = tmp_path / "reference_predictions.npz"
    with np.load(source_reference, allow_pickle=False) as stored:
        selected = {
            key: stored[key]
            for key in stored.files
            if key.startswith(f"{ledger.task_id}__")
        }
    np.savez_compressed(task_reference, **selected)
    report = verify_predictions(tmp_path, task_reference)
    assert report.task_id.unique().tolist() == [ledger.task_id]


def test_subset_verification_uses_selected_reference_task(tmp_path):
    settings_path = ROOT / "configs" / "main_experiment.json"
    config = load_config(settings_path)
    import json
    import numpy as np

    threshold_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    ledger = load_ledger(
        ROOT / "data" / "processed" / "snnrice6ma_rice_chen" / "evidence_ledger.npz"
    )
    _, arrays = run_task(ledger, config, threshold_settings)
    task_directory = tmp_path / ledger.task_id
    task_directory.mkdir()
    np.savez_compressed(task_directory / "predictions.npz", **arrays)
    report = verify_predictions(
        tmp_path,
        ROOT / "results" / "reference" / "reference_predictions.npz",
        task_ids={ledger.task_id},
    )
    assert report.task_id.unique().tolist() == [ledger.task_id]
