import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(1040, 2)

    def forward(self, x, edge_index, batch):
        h = self.lin(x)
        out = torch_geometric.nn.pool.global_mean_pool(h, batch)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    trainer.train_and_validate(Model, "linear")
