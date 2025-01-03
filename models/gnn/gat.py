from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class GATBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module):
        super().__init__()
        # 4 concatenated heads with 256 features => output dimension = 1024
        self.att = torch_geometric.nn.GATConv(1024, 256, 4)
        self.dropout = torch.nn.Dropout(0.5)
        self.act = act

    def forward(self, x, edge_index):
        return self.act(self.dropout(self.att(x, edge_index)))


class Model(torch.nn.Module):
    def __init__(self, num_blocks: int, act: torch.nn.Module):
        super().__init__()
        self.blocks = torch.nn.ModuleList([GATBlock(act) for _ in range(num_blocks)])
        self.readout = torch.nn.Linear(1024, 2)

    def _make_block(self):
        att = torch_geometric.nn.GATConv(1024, 256, 4)
        norm = torch.nn.BatchNorm1d(1024)
        return torch.nn.Sequential(att, norm, self.act)

    def forward(self, x, edge_index, batch):
        for block in self.blocks:
            h = block(x, edge_index)
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    for act in (torch.nn.ReLU(), torch.nn.Identity()):
        trainer.train_and_validate(
            partial(Model, 3, act=act), f"gat_{str(act).split('(')[0]}", 64, 200
        )
