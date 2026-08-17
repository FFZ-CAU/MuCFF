from pathlib import Path

import pandas as pd

from mucff.ledger import discover_ledgers, load_ledger


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_and_ledgers_are_consistent():
    manifest = pd.read_csv(ROOT / "data" / "ledger_manifest.csv").set_index(
        "run_task_id"
    )
    paths = discover_ledgers(ROOT / "data" / "ledgers")
    assert len(paths) == len(manifest) == 21
    for path in paths:
        ledger = load_ledger(path)
        row = manifest.loc[ledger.task_id]
        assert ledger.y_oof.size == row.oof_samples
        assert ledger.y_eval.size == row.evaluation_samples
        assert ledger.n_sources == row.source_count


def test_source_metadata_matches_every_ledger():
    metadata = pd.read_csv(ROOT / "data" / "source_metadata.csv").set_index(
        "source_name"
    )
    for path in discover_ledgers(ROOT / "data" / "ledgers"):
        ledger = load_ledger(path)
        assert set(ledger.source_ids).issubset(metadata.index)
        expected_families = tuple(
            metadata.loc[source_id, "source_family"] for source_id in ledger.source_ids
        )
        assert ledger.source_families == expected_families


def test_primary_and_auxiliary_roles_are_explicit():
    manifest = pd.read_csv(ROOT / "data" / "ledger_manifest.csv")
    primary = manifest.loc[manifest.study_role.eq("Primary")]
    auxiliary = manifest.loc[manifest.study_role.eq("Auxiliary")]
    assert primary.evaluation_task.nunique() == 10
    assert auxiliary.run_task_id.tolist() == [
        "ienhancer_recognition",
        "ienhancer_strength",
    ]
