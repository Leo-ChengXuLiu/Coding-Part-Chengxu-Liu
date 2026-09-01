"""Sparse node/edge multi-task GNN with event context."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    node_dim: int
    edge_dim: int
    hidden_dim: int = 128
    message_layers: int = 4
    dropout: float = 0.05
    use_edge_gate: bool = False
    standardize_edges: bool = False
    log_last_edge_feature: bool = False

    def __post_init__(self) -> None:
        if min(self.node_dim, self.edge_dim, self.hidden_dim, self.message_layers) < 1:
            raise ValueError("All model dimensions and layer counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class GraphBatch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    batch_index: torch.Tensor

    @property
    def number_of_events(self) -> int:
        return int(self.batch_index.max()) + 1 if self.batch_index.numel() else 0

    def validate(self) -> None:
        if self.node_features.ndim != 2:
            raise ValueError("node_features must have shape [nodes, features]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if self.edge_features.ndim != 2:
            raise ValueError("edge_features must have shape [edges, features]")
        if self.edge_index.shape[1] != self.edge_features.shape[0]:
            raise ValueError("edge_index and edge_features contain different edge counts")
        if self.batch_index.ndim != 1 or len(self.batch_index) != len(self.node_features):
            raise ValueError("batch_index must assign every node to one event")
        if self.edge_index.numel() and (
            int(self.edge_index.min()) < 0
            or int(self.edge_index.max()) >= len(self.node_features)
        ):
            raise ValueError("edge_index references a missing node")

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            self.node_features.to(device),
            self.edge_index.to(device),
            self.edge_features.to(device),
            self.batch_index.to(device),
        )


@dataclass
class ModelOutput:
    top_logits: torch.Tensor
    charm_logits: torch.Tensor
    same_top_logits: torch.Tensor
    node_embeddings: torch.Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class MessagePassLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float,
        use_edge_gate: bool,
    ) -> None:
        super().__init__()
        relation_dim = 2 * hidden_dim + edge_dim
        self.message = _mlp(relation_dim, hidden_dim, hidden_dim, dropout)
        self.gate = (
            nn.Sequential(
                nn.Linear(relation_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if use_edge_gate
            else None
        )
        self.update = _mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm(nodes + self.update(torch.zeros_like(nodes)))

        left, right = edge_index
        center = torch.cat((left, right))
        neighbor = torch.cat((right, left))
        directed_edges = torch.cat((edge_features, edge_features), dim=0)
        relation = torch.cat(
            (nodes[center], nodes[neighbor], directed_edges), dim=1
        )
        messages = self.message(relation)
        aggregate = messages.new_zeros(nodes.shape)
        denominator = messages.new_zeros((len(nodes), 1))

        if self.gate is None:
            aggregate.index_add_(0, center, messages)
            denominator.index_add_(0, center, messages.new_ones((len(center), 1)))
        else:
            gates = self.gate(relation).sigmoid()
            aggregate.index_add_(0, center, gates * messages)
            denominator.index_add_(0, center, gates)

        aggregate = aggregate / denominator.clamp_min(1.0e-8)
        return self.norm(nodes + self.update(aggregate))


def _event_context(
    nodes: torch.Tensor,
    batch_index: torch.Tensor,
    number_of_events: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = nodes.new_zeros((number_of_events, nodes.shape[1]))
    sums.index_add_(0, batch_index, nodes)
    counts = torch.bincount(batch_index, minlength=number_of_events).to(nodes.dtype)
    means = sums / counts[:, None].clamp_min(1.0)

    maxima = torch.full_like(sums, -torch.inf)
    maxima.scatter_reduce_(
        0,
        batch_index[:, None].expand_as(nodes),
        nodes,
        reduce="amax",
        include_self=True,
    )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return means, maxima


class SparseMultiTaskGNN(nn.Module):
    """Shared sparse backbone with top-node, charm-node, and same-top-edge heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("edge_mean", torch.zeros(config.edge_dim))
        self.register_buffer("edge_scale", torch.ones(config.edge_dim))
        self.node_encoder = nn.Sequential(
            _mlp(config.node_dim, config.hidden_dim, config.hidden_dim, config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )
        self.message_layers = nn.ModuleList(
            MessagePassLayer(
                config.hidden_dim,
                config.edge_dim,
                config.dropout,
                config.use_edge_gate,
            )
            for _ in range(config.message_layers)
        )
        node_head_dim = 3 * config.hidden_dim
        self.top_head = nn.Sequential(
            _mlp(node_head_dim, config.hidden_dim, config.hidden_dim, config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.charm_head = nn.Sequential(
            _mlp(node_head_dim, config.hidden_dim, config.hidden_dim, config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.same_top_head = nn.Sequential(
            _mlp(
                2 * config.hidden_dim + config.edge_dim,
                config.hidden_dim,
                config.hidden_dim,
                config.dropout,
            ),
            nn.Linear(config.hidden_dim, 1),
        )

    def set_edge_scaler(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        mean = torch.as_tensor(mean, dtype=self.edge_mean.dtype, device=self.edge_mean.device)
        scale = torch.as_tensor(scale, dtype=self.edge_scale.dtype, device=self.edge_scale.device)
        if mean.shape != self.edge_mean.shape or scale.shape != self.edge_scale.shape:
            raise ValueError("Edge scaler dimensions do not match edge_dim")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Edge scaler must be finite with positive scales")
        self.edge_mean.copy_(mean)
        self.edge_scale.copy_(scale)

    def _transform_edges(self, edge_features: torch.Tensor) -> torch.Tensor:
        transformed = edge_features
        if self.config.log_last_edge_feature and edge_features.numel():
            transformed = torch.cat(
                (edge_features[:, :-1], torch.log1p(edge_features[:, -1:])), dim=1
            )
        if self.config.standardize_edges:
            transformed = (transformed - self.edge_mean) / self.edge_scale
        return transformed

    def forward(self, batch: GraphBatch) -> ModelOutput:
        batch.validate()
        if batch.node_features.shape[1] != self.config.node_dim:
            raise ValueError("node_features dimension does not match ModelConfig")
        if batch.edge_features.shape[1] != self.config.edge_dim:
            raise ValueError("edge_features dimension does not match ModelConfig")

        edges = self._transform_edges(batch.edge_features)
        nodes = self.node_encoder(batch.node_features)
        for layer in self.message_layers:
            nodes = layer(nodes, batch.edge_index, edges)

        mean, maximum = _event_context(
            nodes, batch.batch_index, batch.number_of_events
        )
        context = torch.cat(
            (nodes, mean[batch.batch_index], maximum[batch.batch_index]), dim=1
        )
        top_logits = self.top_head(context).squeeze(1)
        charm_logits = self.charm_head(context).squeeze(1)

        if batch.edge_index.numel():
            left, right = batch.edge_index
            symmetric_pair = torch.cat(
                (nodes[left] + nodes[right], (nodes[left] - nodes[right]).abs(), edges),
                dim=1,
            )
            same_top_logits = self.same_top_head(symmetric_pair).squeeze(1)
        else:
            same_top_logits = nodes.new_empty(0)

        return ModelOutput(top_logits, charm_logits, same_top_logits, nodes)


def multitask_loss(
    output: ModelOutput,
    top_labels: torch.Tensor,
    charm_labels: torch.Tensor,
    same_top_labels: torch.Tensor,
    *,
    top_weight: float = 1.0,
    charm_weight: float = 1.0,
    edge_weight: float = 1.0,
    charm_pos_weight: torch.Tensor | None = None,
    edge_pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted sum of the three binary classification losses."""

    top = F.binary_cross_entropy_with_logits(output.top_logits, top_labels.float())
    charm = F.binary_cross_entropy_with_logits(
        output.charm_logits,
        charm_labels.float(),
        pos_weight=charm_pos_weight,
    )
    edge = (
        F.binary_cross_entropy_with_logits(
            output.same_top_logits,
            same_top_labels.float(),
            pos_weight=edge_pos_weight,
        )
        if same_top_labels.numel()
        else output.same_top_logits.sum() * 0.0
    )
    return top_weight * top + charm_weight * charm + edge_weight * edge
