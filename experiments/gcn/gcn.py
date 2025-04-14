from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.nn  # type: ignore
from tqdm import tqdm

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class GCNBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module, feat_dim: int):
        super().__init__()
        self.conv = torch_geometric.nn.GCNConv(feat_dim, feat_dim)
        self.bn = torch.nn.BatchNorm1d(feat_dim)
        self.act = act

    def forward(self, prev, x, edge_index):
        return self.act(self.bn(prev + self.conv(x, edge_index)))


class Model(torch.nn.Module):
    def __init__(
        self, num_blocks: int, feat_dim: int, act: torch.nn.Module = torch.nn.ELU()
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [GCNBlock(act, feat_dim) for _ in range(num_blocks)]
        )
        self.readout = torch.nn.Linear(feat_dim, 2)

    def forward(self, x, edge_index, batch):
        h = x
        prev_h = torch.zeros_like(x)
        for block in self.blocks:
            block_out = block(prev_h, h, edge_index)
            prev_h = h
            h = block_out
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    for depth in tqdm([1, 2, 3, 4, 5, 6]):
        trainer.train_and_validate(
            partial(Model, depth),
            f"gcn_d{depth}",
        )
