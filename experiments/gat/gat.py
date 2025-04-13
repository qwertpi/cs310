from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.nn  # type: ignore
from tqdm import tqdm

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class GATBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module, num_heads: int, feat_dim: int):
        super().__init__()
        if not (feat_dim / num_heads).is_integer():
            raise ValueError
        self.att = torch_geometric.nn.GATConv(
            feat_dim, feat_dim // num_heads, num_heads
        )
        self.bn = torch.nn.BatchNorm1d(feat_dim)
        self.act = act

    def forward(self, prev, x, edge_index):
        return self.act(self.bn(prev + self.att(x, edge_index)))


class Model(torch.nn.Module):
    def __init__(
        self, num_blocks: int, num_heads: int, act: torch.nn.Module, feat_dim: int
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [GATBlock(act, num_heads, feat_dim) for _ in range(num_blocks)]
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
    # The only common factors of 1024 and 1040
    for head_count in tqdm([1, 2, 4, 8, 16]):
        trainer.train_and_validate(
            partial(Model, 3, head_count, torch.nn.ELU()),
            f"gat_h{head_count}",
        )
