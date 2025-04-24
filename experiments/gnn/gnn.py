from functools import partial
from typing import Optional
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.nn  # type: ignore

from GNNModelTrainer import DataSource, GNNModelTrainer  # type: ignore

class GNNBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.lin(x)))


class Model(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_blocks: int = 2,
        internal_dim: Optional[int] = None,
        act: torch.nn.Module = torch.nn.ELU(),
    ):
        super().__init__()
        internal_dim = internal_dim if internal_dim else feat_dim
        self.blocks = torch.nn.ModuleList(
            [GNNBlock(act, feat_dim, internal_dim)]
            + [GNNBlock(act, internal_dim, internal_dim) for _ in range(num_blocks - 1)]
        )
        self.readout = torch.nn.Linear(internal_dim, 2)

    def forward(self, x, edge_index, batch):
        h = x
        for block in self.blocks:
            h = block(h)
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer(DataSource.TCGA)
    trainer.train_and_validate(partial(Model, num_blocks=3), "tcga_gnn")
    trainer = GNNModelTrainer(DataSource.ABCTB)
    trainer.train_and_validate(partial(Model, num_blocks=3), "abctb_gnn")

