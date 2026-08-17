import numpy as np
import pytest

from mucff.attention_baselines import (
    crossfit_qkv_attention,
    fit_score_transform,
    transform_score_tokens,
)
from mucff.fusion import MuCFFConfig


def test_score_token_transform_uses_fitted_reference():
    scores = np.asarray([[0.1, 0.8], [0.4, 0.6], [0.9, 0.2]], dtype=float)
    transform = fit_score_transform(scores, 1e-5)
    tokens = transform_score_tokens(np.asarray([[0.5, 0.7]]), transform, 1e-5)
    assert tokens.shape == (1, 2, 4)
    assert np.isfinite(tokens).all()
    assert tokens[0, 0, 1] == 2.0 / 3.0 - 0.5


def test_qkv_attention_crossfit_is_deterministic():
    pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    labels = np.tile(np.asarray([0, 1]), 40)
    signal = labels[:, None] * 0.35 + rng.normal(0.0, 0.16, size=(80, 5))
    scores = np.clip(0.32 + signal, 0.01, 0.99)
    eval_labels = np.tile(np.asarray([0, 1]), 10)
    eval_signal = eval_labels[:, None] * 0.35 + rng.normal(0.0, 0.16, size=(20, 5))
    eval_scores = np.clip(0.32 + eval_signal, 0.01, 0.99)
    config = MuCFFConfig(
        outer_folds=2,
        attention_dimension=8,
        attention_heads=2,
        attention_hidden_dimension=8,
        attention_batch_size=32,
        attention_max_epochs=2,
        attention_patience=1,
        attention_threads=1,
    )
    first = crossfit_qkv_attention(
        scores, labels, eval_scores, "synthetic", "anchor", config, "cpu"
    )
    second = crossfit_qkv_attention(
        scores, labels, eval_scores, "synthetic", "anchor", config, "cpu"
    )
    assert first[0].shape == labels.shape
    assert first[1].shape == eval_labels.shape
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])
