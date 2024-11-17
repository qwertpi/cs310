import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch_geometric.nn.GCNConv(1024, 2)
        self.add = torch.nn.Linear(1, 1, bias=False)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x, edge_index, batch):
        h = self.conv(x, edge_index)
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        # out = self.softmax(h)
        return h


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    # I.e. all the data in one batch
    trainer.train_and_validate(Model, "linear", 1024, 50)
