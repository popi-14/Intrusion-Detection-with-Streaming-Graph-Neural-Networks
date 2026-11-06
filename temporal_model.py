"""A small recurrent graph classifier with per-node temporal memory."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class TemporalGraphClassifier(nn.Module):
    """Aggregate incoming edge messages and update node memory once per window.

    The memory is intentionally detached between windows (truncated temporal
    backpropagation). Call :meth:`reset_memory` between independent streams.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        class_count: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(node_feature_dim, edge_feature_dim, class_count, hidden_dim) < 1:
            raise ValueError("Kích thước đặc trưng, lớp và hidden phải dương.")
        self.hidden_dim = hidden_dim
        self.edge_feature_dim = edge_feature_dim
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, class_count),
        )
        self.memory: dict[str, torch.Tensor] = {}

    def reset_memory(self) -> None:
        """Clear state before starting an independent event stream."""
        self.memory.clear()

    def forward(
        self,
        node_ids: Sequence[str],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index phải có shape [2, số cạnh].")
        if edge_features.shape != (edge_index.shape[1], self.edge_feature_dim):
            raise ValueError("edge_features không khớp số cạnh hoặc cấu hình model.")
        if node_features.shape != (len(node_ids), self.node_encoder.in_features):
            raise ValueError("node_features không khớp danh sách node hoặc cấu hình model.")
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= len(node_ids)
        ):
            raise ValueError("edge_index tham chiếu node nằm ngoài snapshot.")

        device = node_features.device
        node_state = torch.stack(
            [
                self.memory.get(
                    node_id,
                    torch.zeros(self.hidden_dim, device=device),
                ).to(device)
                for node_id in node_ids
            ]
        )
        encoded_nodes = torch.tanh(self.node_encoder(node_features))
        source, target = edge_index[0].long(), edge_index[1].long()

        aggregated = torch.zeros_like(node_state)
        if source.numel():
            messages = self.message(
                torch.cat(
                    (node_state[source], node_state[target], edge_features), dim=1
                )
            )
            aggregated.index_add_(0, target, messages)
            counts = torch.zeros(
                len(node_ids), 1, dtype=node_features.dtype, device=device
            )
            counts.index_add_(
                0,
                target,
                torch.ones(len(target), 1, dtype=node_features.dtype, device=device),
            )
            aggregated = aggregated / counts.clamp_min(1.0)

        next_state = self.memory_update(
            aggregated + encoded_nodes, node_state
        )
        if source.numel():
            logits = self.classifier(
                torch.cat(
                    (next_state[source], next_state[target], edge_features), dim=1
                )
            )
        else:
            logits = node_features.new_empty((0, self.classifier[-1].out_features))
        self.memory.update(
            {
                node_id: next_state[index].detach()
                for index, node_id in enumerate(node_ids)
            }
        )
        return logits
