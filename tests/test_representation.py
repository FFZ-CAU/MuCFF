import numpy as np

from mucff.representation import aligned_state, mucff_state, residual_state, select_anchor


def test_state_dimensions_and_anchor_selection():
    scores = np.asarray(
        [
            [0.10, 0.45, 0.75],
            [0.20, 0.55, 0.65],
            [0.80, 0.40, 0.35],
            [0.90, 0.60, 0.25],
        ],
        dtype=float,
    )
    labels = np.asarray([0, 0, 1, 1])
    anchor = select_anchor(scores, labels)
    assert anchor == 0
    assert aligned_state(scores).shape == (4, 14)
    assert residual_state(scores, anchor).shape == (4, 12)
    assert mucff_state(scores, anchor).shape == (4, 26)


def test_anchor_residual_is_zero():
    scores = np.asarray([[0.2, 0.7], [0.8, 0.3]], dtype=float)
    residual = residual_state(scores, anchor_index=0)
    source_count = scores.shape[1]
    for offset in (0, source_count, 2 * source_count, 3 * source_count):
        assert np.allclose(residual[:, offset], 0.0)
