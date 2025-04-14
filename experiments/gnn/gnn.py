from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.nn  # type: ignore
from tqdm import tqdm

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class GNNBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module, feat_dim: int):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(feat_dim, feat_dim)
        self.bn = torch.nn.BatchNorm1d(feat_dim)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.lin(x)))


class Model(torch.nn.Module):
    def __init__(
        self, num_blocks: int, feat_dim: int, act: torch.nn.Module = torch.nn.ELU()
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [GNNBlock(act, feat_dim) for _ in range(num_blocks)]
        )
        self.readout = torch.nn.Linear(feat_dim, 2)

    def forward(self, x, edge_index, batch):
        h = x
        for block in self.blocks:
            h = block(h)
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    for depth in tqdm([1, 2, 3, 4, 5, 6]):
        trainer.train_and_validate(
            partial(Model, depth),
            f"gnn_d{depth}",
        )
