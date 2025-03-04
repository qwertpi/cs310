from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class GCNBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module):
        super().__init__()
        self.conv = torch_geometric.nn.GCNConv(1024, 1024)
        self.dropout = torch.nn.Dropout(0.5)
        self.act = act

    def forward(self, x, edge_index):
        return self.act(
            self.dropout(torch.sum(torch.tensor([x, self.conv(x, edge_index)])))
        )


class Model(torch.nn.Module):
    def __init__(self, num_blocks: int, act: torch.nn.Module):
        super().__init__()
        self.blocks = torch.nn.ModuleList([GCNBlock(act) for _ in range(num_blocks)])
        self.readout = torch.nn.Linear(1024, 2)

    def forward(self, x, edge_index, batch):
        h = x
        for block in self.blocks:
            h = block(h, edge_index)
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    trainer.train_and_validate(
        partial(Model, 2, act=torch.nn.Identity()),
        "gcn",
    )
