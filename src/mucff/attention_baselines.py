"""Cross-fitted QKV attention baselines for score-level fusion."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from .fusion import MuCFFConfig, stable_seed
from .representation import clip_prob, logit, select_anchor


@dataclass(frozen=True)
class ScoreTransform:
    sorted_probabilities: np.ndarray


def fit_score_transform(scores: np.ndarray, epsilon: float) -> ScoreTransform:
    probabilities = clip_prob(scores, epsilon)
    return ScoreTransform(np.sort(probabilities, axis=0))


def transform_score_tokens(
    scores: np.ndarray,
    transform: ScoreTransform,
    epsilon: float,
) -> np.ndarray:
    probabilities = clip_prob(scores, epsilon)
    if probabilities.shape[1] != transform.sorted_probabilities.shape[1]:
        raise ValueError("Score transform and input have different source counts.")
    ranks = np.empty_like(probabilities, dtype=np.float64)
    denominator = max(transform.sorted_probabilities.shape[0], 1)
    for source_index in range(probabilities.shape[1]):
        ranks[:, source_index] = np.searchsorted(
            transform.sorted_probabilities[:, source_index],
            probabilities[:, source_index],
            side="right",
        ) / denominator
    logits = np.clip(logit(probabilities, epsilon), -8.0, 8.0) / 8.0
    confidence = 2.0 * np.abs(probabilities - 0.5)
    return np.stack(
        [probabilities - 0.5, ranks - 0.5, logits, confidence], axis=-1
    ).astype(np.float32)


def flattened_token_state(tokens: np.ndarray) -> np.ndarray:
    probabilities = tokens[:, :, 0] + 0.5
    summaries = np.column_stack(
        [
            probabilities.mean(axis=1),
            probabilities.std(axis=1),
            probabilities.min(axis=1),
            probabilities.max(axis=1),
            np.ptp(probabilities, axis=1),
        ]
    )
    return np.hstack([tokens.reshape(tokens.shape[0], -1), summaries]).astype(np.float32)


def _torch_modules():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise ImportError(
            "QKV baselines require PyTorch; install the 'sources' optional dependency."
        ) from error
    return torch, nn, DataLoader, TensorDataset


def _make_model(
    source_count: int,
    query_mode: str,
    anchor_index: int,
    base_coefficients: np.ndarray,
    base_intercept: float,
    config: MuCFFConfig,
):
    torch, nn, _, _ = _torch_modules()
    if query_mode not in {"self", "anchor"}:
        raise ValueError(f"Unknown QKV query mode: {query_mode}")
    if config.attention_dimension % config.attention_heads:
        raise ValueError("Attention dimension must be divisible by the number of heads.")

    class ScoreAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = config.attention_dimension
            self.query_mode = query_mode
            self.anchor_index = anchor_index
            self.token_projection = nn.Linear(4, dimension)
            self.source_embedding = nn.Embedding(source_count, dimension)
            self.token_norm = nn.LayerNorm(dimension)
            if self.query_mode == "self":
                self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
                nn.init.normal_(self.cls_token, std=0.02)
            else:
                self.register_parameter("cls_token", None)
            self.attention = nn.MultiheadAttention(
                dimension,
                config.attention_heads,
                dropout=config.attention_dropout,
                batch_first=True,
            )
            self.output_norm = nn.LayerNorm(dimension)
            self.decision = nn.Sequential(
                nn.Linear(dimension, config.attention_hidden_dimension),
                nn.GELU(),
                nn.Dropout(config.attention_dropout),
                nn.Linear(config.attention_hidden_dimension, 1),
            )
            self.linear_skip = nn.Linear(4 * source_count + 5, 1)
            with torch.no_grad():
                self.linear_skip.weight.copy_(
                    torch.from_numpy(base_coefficients.astype(np.float32))[None, :]
                )
                self.linear_skip.bias.fill_(float(base_intercept))
                nn.init.zeros_(self.decision[-1].weight)
                nn.init.zeros_(self.decision[-1].bias)
            self.linear_skip.requires_grad_(False)

        def forward(self, tokens):
            source_index = torch.arange(tokens.shape[1], device=tokens.device)
            encoded = self.token_projection(tokens) + self.source_embedding(source_index)[None, :, :]
            encoded = self.token_norm(encoded)
            if self.query_mode == "self":
                cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
                sequence = torch.cat([cls_token, encoded], dim=1)
                attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
                pooled = self.output_norm(attended + sequence)[:, 0, :]
            else:
                query = encoded[:, self.anchor_index : self.anchor_index + 1, :]
                attended, _ = self.attention(query, encoded, encoded, need_weights=False)
                pooled = self.output_norm(attended + query)[:, 0, :]
            interaction_logit = self.decision(pooled).squeeze(1)
            probabilities = tokens[:, :, 0] + 0.5
            summaries = torch.stack(
                [
                    probabilities.mean(dim=1),
                    probabilities.std(dim=1, unbiased=False),
                    probabilities.amin(dim=1),
                    probabilities.amax(dim=1),
                    probabilities.amax(dim=1) - probabilities.amin(dim=1),
                ],
                dim=1,
            )
            linear_state = torch.cat([tokens.flatten(start_dim=1), summaries], dim=1)
            linear_logit = self.linear_skip(linear_state).squeeze(1)
            return interaction_logit + linear_logit

    return ScoreAttention()


def _predict(model, tokens: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    torch, _, DataLoader, TensorDataset = _torch_modules()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tokens)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    values = []
    with torch.no_grad():
        for (batch,) in loader:
            values.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return clip_prob(np.concatenate(values)).astype(np.float32)


def _fit_fold(
    train_tokens: np.ndarray,
    train_labels: np.ndarray,
    validation_tokens: np.ndarray,
    validation_labels: np.ndarray,
    source_count: int,
    query_mode: str,
    anchor_index: int,
    seed: int,
    config: MuCFFConfig,
    device: str,
):
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, config.attention_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setting only before inter-op work begins.
        pass
    base_state = flattened_token_state(train_tokens)
    scaler = StandardScaler().fit(base_state)
    linear_base = LogisticRegression(
        C=0.10,
        max_iter=1800,
        class_weight="balanced",
        random_state=seed,
    )
    linear_base.fit(scaler.transform(base_state), train_labels)
    effective_coefficients = linear_base.coef_[0] / scaler.scale_
    effective_intercept = float(
        linear_base.intercept_[0] - np.dot(effective_coefficients, scaler.mean_)
    )
    model = _make_model(
        source_count,
        query_mode,
        anchor_index,
        effective_coefficients,
        effective_intercept,
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.attention_learning_rate,
        weight_decay=config.attention_weight_decay,
    )
    positives = max(int(train_labels.sum()), 1)
    negatives = max(int(train_labels.size - train_labels.sum()), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    )
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(train_tokens), torch.from_numpy(train_labels.astype(np.float32))
    )
    loader = DataLoader(
        dataset,
        batch_size=config.attention_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    initial_probability = _predict(
        model, validation_tokens, config.attention_batch_size, device
    )
    best_auc = roc_auc_score(validation_labels, initial_probability)
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    for _ in range(config.attention_max_epochs):
        model.train()
        for batch_tokens, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_tokens.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
        validation_probability = _predict(
            model, validation_tokens, config.attention_batch_size, device
        )
        validation_auc = roc_auc_score(validation_labels, validation_probability)
        if validation_auc > best_auc + 1e-5:
            best_auc = validation_auc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.attention_patience:
                break
    model.load_state_dict(best_state)
    return model


def crossfit_qkv_attention(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    eval_scores: np.ndarray,
    task_id: str,
    query_mode: str,
    config: MuCFFConfig,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(
        config.outer_folds,
        shuffle=True,
        random_state=stable_seed(config.seed_base, task_id, "fusion_benchmark_common_cv"),
    )
    oof_probability = np.zeros(labels.size, dtype=np.float32)
    eval_probabilities = []
    for fold, (outer_train, outer_validation) in enumerate(splitter.split(oof_scores, labels)):
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.attention_validation_fraction,
            random_state=stable_seed(config.seed_base, task_id, query_mode, fold, "inner"),
        )
        fit_relative, early_relative = next(
            inner_splitter.split(oof_scores[outer_train], labels[outer_train])
        )
        fit_index = outer_train[fit_relative]
        early_index = outer_train[early_relative]
        transform = fit_score_transform(oof_scores[outer_train], config.probability_epsilon)
        fit_tokens = transform_score_tokens(
            oof_scores[fit_index], transform, config.probability_epsilon
        )
        early_tokens = transform_score_tokens(
            oof_scores[early_index], transform, config.probability_epsilon
        )
        validation_tokens = transform_score_tokens(
            oof_scores[outer_validation], transform, config.probability_epsilon
        )
        evaluation_tokens = transform_score_tokens(
            eval_scores, transform, config.probability_epsilon
        )
        anchor_index = select_anchor(
            oof_scores[fit_index], labels[fit_index], config.probability_epsilon
        )
        seed = stable_seed(config.model_seed, task_id, query_mode, fold)
        model = _fit_fold(
            fit_tokens,
            labels[fit_index],
            early_tokens,
            labels[early_index],
            oof_scores.shape[1],
            query_mode,
            anchor_index,
            seed,
            config,
            device,
        )
        oof_probability[outer_validation] = _predict(
            model, validation_tokens, config.attention_batch_size, device
        )
        eval_probabilities.append(
            _predict(model, evaluation_tokens, config.attention_batch_size, device)
        )
    return (
        clip_prob(oof_probability, config.probability_epsilon).astype(np.float32),
        clip_prob(
            np.mean(eval_probabilities, axis=0), config.probability_epsilon
        ).astype(np.float32),
    )


def run_qkv_attention_baselines(
    ledger,
    config: MuCFFConfig,
    device: str = "cpu",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "source_self_attention": crossfit_qkv_attention(
            ledger.oof_scores,
            ledger.y_oof,
            ledger.eval_scores,
            ledger.task_id,
            "self",
            config,
            device,
        ),
        "anchor_query_qkv": crossfit_qkv_attention(
            ledger.oof_scores,
            ledger.y_oof,
            ledger.eval_scores,
            ledger.task_id,
            "anchor",
            config,
            device,
        ),
    }
