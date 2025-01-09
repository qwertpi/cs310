from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore
from tqdm.contrib.itertools import product

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class GCNBlock(torch.nn.Module):
    def __init__(self, act: torch.nn.Module):
        super().__init__()
        self.att = torch_geometric.nn.GCNConv(1024, 1024)
        self.dropout = torch.nn.Dropout(0.5)
        self.act = act

    def forward(self, x, edge_index):
        return self.act(self.dropout(self.att(x, edge_index)))


class Model(torch.nn.Module):
    def __init__(self, num_blocks: int, act: torch.nn.Module):
        super().__init__()
        self.blocks = torch.nn.ModuleList([GCNBlock(act) for _ in range(num_blocks)])
        self.readout = torch.nn.Linear(1024, 2)

    def forward(self, x, edge_index, batch):
        for block in self.blocks:
            h = block(x, edge_index)
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    for num_blocks, decay in product([1, 2, 3, 4], [1e-1, 1e-2, 1e-3, 0]):
        trainer.train_and_validate(
            partial(Model, num_blocks, act=torch.nn.Identity()),
            f"gcn_b{num_blocks}_w{decay}",
            decay,
        )
