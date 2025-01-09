# Based on https://github.com/engrodawood/HistBiases/blob/1aacf8ae86acfea5d29d55096a8af2b435ba99af/application/MLModels/SlideGraph%5E%7Binf%7D/model/gnn.py

from functools import partial
import sys

sys.path.insert(0, "..")

import torch
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

from GNNModelTrainer import GNNModelTrainer  # type: ignore


class Subnet(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return self.bn(self.lin(x))


class Block(torch.nn.Module):
    def __init__(self, subblock: torch.nn.Module):
        super().__init__()
        self.subblock = subblock
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, edge_index, batch):
        if isinstance(self.subblock, torch_geometric.nn.MessagePassing):
            h = self.subblock(x, edge_index)
        else:
            h = self.subblock(x)
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        return self.dropout(h)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        subblocks = torch.nn.ModuleList(
            [Subnet(1024, 1024), self._make_edgeconv(), self._make_edgeconv()]
        )
        self.blocks = torch.nn.ModuleList([Block(subblock) for subblock in subblocks])
        self.readout = torch.nn.Linear(1024, 2)

    def _make_edgeconv(self):
        return torch_geometric.nn.EdgeConv(Subnet(2048, 1024), aggr="max")

    def forward(self, x, edge_index, batch):
        acc = torch.stack(
            [block(x, edge_index, batch) for block in self.blocks], dim=0
        ).sum(dim=0)
        return self.readout(acc)


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    trainer.train_and_validate(
        partial(Model),
        "econv_nolocalproj",
        1e-2,
    )
