from functools import partial
from math import log2
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.nn  # type: ignore
from tqdm import tqdm

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class GCNBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = torch_geometric.nn.GCNConv(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.act = act

    def forward(self, prev, x, edge_index):
        return self.act(self.bn(prev + self.conv(x, edge_index)))


class Model(torch.nn.Module):
    def __init__(
        self,
        internal_dim: int,
        feat_dim: int,
        num_blocks: int = 3,
        act: torch.nn.Module = torch.nn.ELU(),
    ):
        super().__init__()
        self.internal_dim = internal_dim
        self.blocks = torch.nn.ModuleList(
            [GCNBlock(act, feat_dim, internal_dim)]
            + [GCNBlock(act, internal_dim, internal_dim) for _ in range(num_blocks - 1)]
        )
        self.readout = torch.nn.Linear(internal_dim, 2)

    def forward(self, x, edge_index, batch):
        h = x
        prev_h = torch.zeros_like(x)[:, :self.internal_dim]
        for block in self.blocks:
            block_out = block(prev_h, h, edge_index)
            prev_h = h[:, :self.internal_dim]
            h = block_out
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    for width in tqdm([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
        trainer.train_and_validate(
            partial(Model, width),
            f"gcn_w{int(log2(width))}",
        )
