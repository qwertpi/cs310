from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore
from tqdm import tqdm
# from tqdm.contrib.itertools import product

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class Model(torch.nn.Module):
    def __init__(self, act: torch.nn.Module):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(1024, 2)
        self.act = act

    def forward(self, x, edge_index, batch):
        h = self.act(self.lin(x))
        out = torch_geometric.nn.pool.global_mean_pool(h, batch)
        return out


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    for i, act in tqdm(
        [(1, torch.nn.Identity()), (2, torch.nn.PReLU())]
    ):
        trainer.train_and_validate(partial(Model, act), f"linear_a{i}")
