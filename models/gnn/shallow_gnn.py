from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

from GNNModelTrainerCrossLabel import GNNModelTrainer  # type: ignore


class Model(torch.nn.Module):
    def __init__(self, act: torch.nn.Module):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(1024, 1024)
        self.act = act
        self.readout = torch.nn.Linear(1024, 2)

    def forward(self, x, edge_index, batch):
        h = self.lin(x)
        h = self.act(h)
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    for act in (torch.nn.Tanh(), torch.nn.ReLU(), torch.nn.Identity()):
        trainer.train_and_validate(
            partial(Model, act=act), f"gnn_{str(act).split('(')[0]}", 64, 200
        )
