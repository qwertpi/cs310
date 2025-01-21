# Based on https://github.com/engrodawood/HistBiases/blob/1aacf8ae86acfea5d29d55096a8af2b435ba99af/application/MLModels/SlideGraph%5E%7Binf%7D/model/gnn.py

from abc import ABC, abstractmethod
from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore
from GNNModelTrainer import GNNModelTrainer  # type: ignore


class Subnet(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return self.bn(self.lin(x))


class InitialSubnet(Subnet):
    def __init__(self):
        super().__init__(1024, 1024)


class EdgeConvSubnet(Subnet):
    def __init__(self):
        super().__init__(2048, 1024)


class Block(torch.nn.Module):
    def __init__(self, subblock: torch.nn.Module):
        super().__init__()
        self.subblock = subblock
        self.lin = torch_geometric.nn.Linear(1024, 2)
        self.bn = torch.nn.BatchNorm1d(2)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, edge_index, batch):
        if isinstance(self.subblock, torch_geometric.nn.MessagePassing):
            h = self.subblock(x, edge_index)
        else:
            h = self.subblock(x)
        h = self.bn(self.lin(h))
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        return self.dropout(h)


class Model(torch.nn.Module):
    def __init__(self, num_initial_layers: int, num_middle_layers: int):
        super().__init__()
        subblocks = torch.nn.ModuleList(
            [InitialSubnet() for _ in range(num_initial_layers)]
            + [
                torch_geometric.nn.EdgeConv(EdgeConvSubnet(), aggr="max")
                for _ in range(num_middle_layers)
            ]
        )
        self.blocks = torch.nn.ModuleList([Block(subblock) for subblock in subblocks])
        self.readout = torch.nn.Linear(2, 2)

    def forward(self, x, edge_index, batch):
        acc = torch.stack(
            [block(x, edge_index, batch) for block in self.blocks], dim=0
        ).sum(dim=0)
        return self.readout(acc)


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    trainer.train_and_validate(
        partial(Model, 0, 3),
        "econv",
        1e-2,
    )
