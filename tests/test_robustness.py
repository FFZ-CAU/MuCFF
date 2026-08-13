import numpy as np

from mucff.fusion import MuCFFConfig
from mucff.ledger import EvidenceLedger
from mucff.representation import clip_prob, logit, mucff_state, rank_columns
from mucff.robustness import _state_from_components, build_stress_conditions


def test_stress_conditions_include_source_groups_and_random_missingness():
    rng = np.random.default_rng(7)
    ledger = EvidenceLedger(
        task_id="synthetic",
        y_oof=np.tile([0, 1], 20),
        y_eval=np.tile([0, 1], 5),
        oof_scores=rng.uniform(0.1, 0.9, size=(40, 4)),
        eval_scores=rng.uniform(0.1, 0.9, size=(10, 4)),
        source_ids=("a", "b", "c", "d"),
        source_families=(
            "Composition and FCGR",
            "DNABERT-2",
            "RC motif grammar",
            "Cross-fitted meta-evidence",
        ),
    )
    conditions = build_stress_conditions(ledger, MuCFFConfig(), repeats=2)
    names = [condition.name for condition in conditions]
    assert "missing_group:engineered_descriptors" in names
    assert "missing_group:foundation_evidence" in names
    assert names.count("random_missing_20pct") == 2
    assert names.count("random_missing_40pct") == 2
    assert names.count("score_noise_sd_0.05") == 2


def test_cached_state_matches_reference_representation():
    rng = np.random.default_rng(11)
    scores = rng.uniform(0.05, 0.95, size=(31, 6))
    probabilities = clip_prob(scores)
    cached = _state_from_components(
        probabilities, rank_columns(probabilities), logit(probabilities), anchor_index=2
    )
    np.testing.assert_allclose(cached, mucff_state(scores, anchor_index=2), atol=0.0)
