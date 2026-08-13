"""Reverse-complement grammar and cross-source embedding modules."""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
except ImportError as error:
    raise ImportError("Install the 'sources' optional dependency to use neural source modules.") from error


NUCLEOTIDE_CHEMICAL_PROPERTY = torch.tensor(
    [
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.gelu(values + self.network(values))


class MotifGrammarEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 72,
        dropout: float = 0.18,
        biophysical_channels: bool = False,
        position_channels: bool = False,
    ) -> None:
        super().__init__()
        self.biophysical_channels = biophysical_channels
        self.position_channels = position_channels
        self.register_buffer("ncp", NUCLEOTIDE_CHEMICAL_PROPERTY.clone(), persistent=False)
        input_channels = (11 if biophysical_channels else 4) + (2 if position_channels else 0)
        branch_channels = channels // 3
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_channels, branch_channels, kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                )
                for kernel in (7, 11, 19)
            ]
        )
        self.project = nn.Conv1d(branch_channels * 3, channels, 1, bias=False)
        self.blocks = nn.Sequential(
            DilatedResidualBlock(channels, 1, dropout),
            DilatedResidualBlock(channels, 2, dropout),
            DilatedResidualBlock(channels, 4, dropout),
        )
        self.attention = nn.Conv1d(channels, 1, 1)
        self.output_norm = nn.LayerNorm(channels * 3)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        mask = indices < 4
        one_hot = functional.one_hot(indices.long(), num_classes=5)[..., :4].float().transpose(1, 2)
        values = one_hot
        if self.biophysical_channels:
            properties = self.ncp[indices.long()].transpose(1, 2)
            positions = torch.arange(1, indices.shape[1] + 1, device=indices.device, dtype=one_hot.dtype)[None, None, :]
            values = torch.cat([one_hot, properties, one_hot.cumsum(dim=2) / positions], dim=1)
        if self.position_channels:
            coordinate = torch.linspace(-1.0, 1.0, indices.shape[1], device=indices.device)[None, None, :]
            coordinate = coordinate.expand(indices.shape[0], -1, -1)
            values = torch.cat([values, torch.cat([coordinate, coordinate.abs()], dim=1) * mask[:, None]], dim=1)
        values = torch.cat([branch(values) for branch in self.branches], dim=1)
        values = self.blocks(self.project(values))
        attention = self.attention(values).squeeze(1).masked_fill(~mask, -1e4).softmax(dim=1)
        attention_pool = torch.sum(values * attention[:, None], dim=2)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
        mean_pool = (values * mask[:, None]).sum(dim=2) / denominator
        max_pool = values.masked_fill(~mask[:, None], -1e4).amax(dim=2)
        return self.output_norm(torch.cat([attention_pool, mean_pool, max_pool], dim=1))


class ReverseComplementGrammarExpert(nn.Module):
    def __init__(
        self,
        channels: int = 72,
        dropout: float = 0.18,
        biophysical_channels: bool = False,
        position_channels: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = MotifGrammarEncoder(
            channels, dropout, biophysical_channels, position_channels
        )
        representation = channels * 3
        self.head = nn.Sequential(
            nn.LayerNorm(representation * 4),
            nn.Linear(representation * 4, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 1),
        )

    def forward(self, forward: torch.Tensor, reverse_complement: torch.Tensor) -> torch.Tensor:
        forward_embedding = self.encoder(forward)
        reverse_embedding = self.encoder(reverse_complement)
        orientation = forward_embedding - reverse_embedding
        state = torch.cat(
            [
                0.5 * (forward_embedding + reverse_embedding),
                orientation,
                orientation.abs(),
                forward_embedding * reverse_embedding,
            ],
            dim=1,
        )
        return self.head(state).squeeze(1)


class CrossSourceEmbeddingFusion(nn.Module):
    def __init__(self, input_dimensions: list[int], hidden: int = 96, heads: int = 4) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden), nn.GELU())
                for dimension in input_dimensions
            ]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.gate = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.20), nn.Linear(hidden, 1))

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        if len(sources) != len(self.adapters):
            raise ValueError("The number of source tensors does not match the configured adapters.")
        tokens = torch.stack(
            [adapter(source.float()) for adapter, source in zip(self.adapters, sources, strict=True)],
            dim=1,
        )
        tokens = self.encoder(tokens)
        weights = torch.softmax(self.gate(tokens).squeeze(-1), dim=1)
        fused = torch.sum(tokens * weights[:, :, None], dim=1)
        return self.head(fused).squeeze(1)

