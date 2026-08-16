from pathlib import Path

import pandas as pd

from mucff.ledger import discover_ledgers, load_ledger


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_and_ledgers_are_consistent():
    manifest = pd.read_csv(ROOT / "data" / "ledger_manifest.csv").set_index("task_id")
    paths = discover_ledgers(ROOT / "data" / "processed")
    assert len(paths) == len(manifest) == 10
    for path in paths:
        ledger = load_ledger(path)
        row = manifest.loc[ledger.task_id]
        assert ledger.y_oof.size == row.oof_samples
        assert ledger.y_eval.size == row.evaluation_samples
        assert ledger.n_sources == row.source_count


def test_auxiliary_ledgers_use_stable_source_metadata():
    metadata = pd.read_csv(ROOT / "data" / "auxiliary_source_metadata.csv")
    expected_ids = metadata["source_id"].tolist()
    expected_families = metadata["source_family"].tolist()
    paths = discover_ledgers(ROOT / "data" / "auxiliary")
    assert len(paths) == 2
    for path in paths:
        ledger = load_ledger(path)
        assert list(ledger.source_ids) == expected_ids
        assert list(ledger.source_families) == expected_families
