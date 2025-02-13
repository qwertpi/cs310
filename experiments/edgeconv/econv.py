# Based on https://github.com/engrodawood/HistBiases/blob/1aacf8ae86acfea5d29d55096a8af2b435ba99af/application/MLModels/SlideGraph%5E%7Binf%7D/model/gnn.py

from abc import ABC, abstractmethod
from functools import partial
import sys
from typing import Callable

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore
from GNNModelTrainer import GNNModelTrainer  # type: ignore


class Subnet(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(
        self, in_dim: int, out_dim: int, act: Callable[[torch.Tensor], torch.Tensor]
    ):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.lin(x)))


class InitialSubnet(Subnet):
    def __init__(self):
        super().__init__(1024, 1024, torch.nn.functional.gelu)


class EdgeConvSubnet(Subnet):
    def __init__(self):
        super().__init__(2048, 1024, torch.nn.functional.elu)


class PostProccessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(1024, 2)
        self.bn = torch.nn.BatchNorm1d(2)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, batch):
        h = x
        h = torch.nn.functional.elu(self.bn(self.lin(h)))
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        return self.dropout(h)


class Model(torch.nn.Module):
    def __init__(self, num_initial_layers: int, num_middle_layers: int):
        super().__init__()
        self.subblocks = torch.nn.ModuleList(
            [InitialSubnet() for _ in range(num_initial_layers)]
            + [
                torch_geometric.nn.EdgeConv(EdgeConvSubnet(), aggr="max")
                for _ in range(num_middle_layers)
            ]
        )
        self.posts = torch.nn.ModuleList([PostProccessing() for _ in self.subblocks])
        self.readout = torch.nn.Linear(2, 2)

    def forward(self, x, edge_index, batch):
        block_outputs = []
        h = x
        for subblock, post in zip(self.subblocks, self.posts):
            if isinstance(subblock, torch_geometric.nn.MessagePassing):
                h = subblock(h, edge_index)
            else:
                h = subblock(h)
            block_outputs.append(post(h, batch))
        return self.readout(torch.stack(block_outputs, dim=0).sum(dim=0))


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    trainer.train_and_validate(
        partial(Model, 1, 3),
        "econv",
    )
