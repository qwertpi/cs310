import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore
from GNNModelTrainer import GNNModelTrainer  # type: ignore


# Removes randomly selected nodes from the graph
class NodeDropout(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x, edge_idx, batch):
        edge_idx, _, node_mask = torch_geometric.utils.dropout.dropout_node(
            edge_idx, self.p, x.size(0)
        )
        return x[node_mask], edge_idx, batch[node_mask]


# Replaces the features of randomly selected nodes with the graph's average
class NodeFeatureDropout(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        mask = torch.nn.functional.dropout(
            torch.ones(x.size(0), device=x.device), p=self.p
        ).bool()
        num_dropped = mask.sum()
        x[mask] = torch.normal(
            mean.expand(num_dropped, -1), std.expand(num_dropped, -1)
        )
        return x


class AttentionAggregation(torch.nn.Module):
    def __init__(
        self, in_channels: int, intermediate_channels: int, gated: bool = False
    ):
        super().__init__()
        self.gated = gated
        self.gate = torch_geometric.nn.Linear(in_channels, intermediate_channels)
        self.projection = torch_geometric.nn.Linear(in_channels, intermediate_channels)
        self.readout = torch_geometric.nn.Linear(intermediate_channels, 1)

    def forward(self, x, batch):
        projected = torch.nn.functional.tanh(self.projection(x))
        if self.gated:
            gate_value = torch.nn.functional.sigmoid(self.gate(x))
        else:
            gate_value = torch.ones_like(projected, device=x.device)
        # Note this is element-wise multiplication not matrix-multiplication
        gated_projected = gate_value * projected
        coefficents = torch.nn.functional.softmax(self.readout(gated_projected), dim=0)
        return torch_geometric.nn.pool.global_add_pool(coefficents * x, batch)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                # The mean number of nodes is 1338; in expectation we will have 67 nodes
                NodeDropout(0.95),
                NodeFeatureDropout(0.75),
                torch_geometric.nn.Linear(1024, 512),
                torch.nn.Dropout(p=0.5),
                torch_geometric.nn.Linear(512, 512),
                torch.nn.Dropout(p=0.5),
                AttentionAggregation(512, intermediate_channels=128),
                torch.nn.Linear(512, 2),
            ]
        )

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            if isinstance(layer, NodeDropout):
                x, edge_idx, batch = layer(x, edge_index, batch)
            elif isinstance(layer, AttentionAggregation):
                x = layer(x, batch)
            else:
                x = layer(x)
        return x


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    trainer.train_and_validate(
        Model,
        "rnet",
    )
