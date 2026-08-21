import numpy as np

from mucff.representation import (
    aligned_state,
    compact_residual_state,
    fit_compact_residual_routing,
    mucff_state,
    residual_state,
    select_anchor,
)


def test_state_dimensions_and_anchor_selection():
    scores = np.asarray(
        [
            [0.10, 0.45, 0.75, 0.30],
            [0.20, 0.55, 0.65, 0.40],
            [0.80, 0.40, 0.35, 0.60],
            [0.90, 0.60, 0.25, 0.70],
        ],
        dtype=float,
    )
    labels = np.asarray([0, 0, 1, 1])
    anchor = select_anchor(scores, labels)
    routing = fit_compact_residual_routing(scores, labels, ("a", "b", "c", "d"))
    assert anchor == 0
    assert aligned_state(scores).shape == (4, 17)
    assert residual_state(scores, anchor).shape == (4, 16)
    assert compact_residual_state(scores, routing).shape == (4, 12)
    assert mucff_state(scores, anchor).shape == (4, 33)


def test_anchor_residual_is_zero():
    scores = np.asarray([[0.2, 0.7], [0.8, 0.3]], dtype=float)
    residual = residual_state(scores, anchor_index=0)
    source_count = scores.shape[1]
    for offset in (0, source_count, 2 * source_count, 3 * source_count):
        assert np.allclose(residual[:, offset], 0.0)
