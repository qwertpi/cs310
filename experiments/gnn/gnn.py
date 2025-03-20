from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

from GNNModelTrainer import GNNModelTrainer  # type: ignore

FEAT_DIM = 1040
class GNNBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(FEAT_DIM, FEAT_DIM)
        self.bn = torch.nn.BatchNorm1d(FEAT_DIM)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.lin(x)))


class Model(torch.nn.Module):
    def __init__(self, num_blocks: int, act: torch.nn.Module):
        super().__init__()
        self.blocks = torch.nn.ModuleList([GNNBlock(act) for _ in range(num_blocks)])
        self.readout = torch.nn.Linear(FEAT_DIM, 2)

    def forward(self, x, edge_index, batch):
        h = x
        for block in self.blocks:
            h = block(h)
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    trainer.train_and_validate(partial(Model, 1, torch.nn.ELU()), "gnn")
