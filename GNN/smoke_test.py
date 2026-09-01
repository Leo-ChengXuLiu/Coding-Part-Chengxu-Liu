"""Minimal forward/backward and reconstruction-to-BDT smoke test."""

from __future__ import annotations

import torch

from src import (
    DecodeConfig,
    GraphBatch,
    ModelConfig,
    ModelOutput,
    SparseMultiTaskGNN,
    decode_event,
    multitask_loss,
)


def main() -> None:
    torch.manual_seed(7)
    node_features = torch.randn(7, 34)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    edge_features = torch.randn(5, 6)
    batch = GraphBatch(
        node_features,
        edge_index,
        edge_features,
        torch.zeros(7, dtype=torch.long),
    )
    model = SparseMultiTaskGNN(
        ModelConfig(node_dim=34, edge_dim=6, hidden_dim=32, message_layers=2)
    )
    output = model(batch)
    loss = multitask_loss(
        output,
        torch.tensor([1, 1, 1, 0, 0, 0, 0]),
        torch.tensor([0, 0, 0, 1, 0, 0, 0]),
        torch.tensor([1, 1, 0, 0, 0]),
    )
    loss.backward()

    scores = ModelOutput(
        top_logits=torch.tensor([6.0, 5.0, 4.0, -4.0, -5.0, -6.0, -7.0]),
        charm_logits=torch.tensor([-5.0, -5.0, -5.0, 6.0, 1.0, 0.0, -1.0]),
        same_top_logits=torch.tensor([6.0, 5.0, -5.0, -5.0, -5.0]),
        node_embeddings=output.node_embeddings.detach(),
    )
    p4 = torch.tensor(
        [
            [120.0, 70.0, 10.0, 80.0],
            [80.0, 35.0, 20.0, 55.0],
            [45.0, 15.0, -8.0, 30.0],
            [90.0, -55.0, -12.0, -60.0],
            [35.0, 15.0, 5.0, 20.0],
            [30.0, -8.0, 3.0, 15.0],
            [20.0, 5.0, 2.0, -8.0],
        ]
    )
    decoded = decode_event(scores, p4, edge_index, DecodeConfig(max_top_nodes=3))
    assert decoded.top_node_indices == (0, 1, 2)
    assert decoded.charm_node_index == 3
    assert decoded.bdt_features["top_group_size"] == 3.0
    assert decoded.bdt_features["top_energy"] == 245.0
    assert len(decoded.bdt_features) == 29
    print("GNN_RECONSTRUCTION_BDT_PACK_SMOKE_PASS")


if __name__ == "__main__":
    main()
