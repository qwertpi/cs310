from functools import partial
from math import log
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class GATBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module, num_heads: int):
        super().__init__()
        if not log(num_heads, 2).is_integer():
            raise ValueError
        self.att = torch_geometric.nn.GATConv(1024, 1024 // num_heads, num_heads)
        self.bn = torch.nn.BatchNorm1d(1024)
        self.act = act

    def forward(self, prev, x, edge_index):
        return self.act(self.bn(prev + self.att(x, edge_index)))


class Model(torch.nn.Module):
    def __init__(self, num_blocks: int, num_heads: int, act: torch.nn.Module):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [GATBlock(act, num_heads) for _ in range(num_blocks)]
        )
        self.readout = torch.nn.Linear(1024, 2)

    def forward(self, x, edge_index, batch):
        h = x
        prev_h = h
        for block in self.blocks:
            block_out = block(prev_h, h, edge_index)
            prev_h = h
            h = block_out
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    trainer.train_and_validate(
        partial(Model, 2, 256, torch.nn.PReLU()),
        "gat",
    )
