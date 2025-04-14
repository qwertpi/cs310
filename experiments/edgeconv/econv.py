# Based on https://github.com/engrodawood/HistBiases/blob/1aacf8ae86acfea5d29d55096a8af2b435ba99af/application/MLModels/SlideGraph%5E%7Binf%7D/model/gnn.py

from abc import ABC, abstractmethod
from functools import partial
import sys
from typing import Callable

sys.path.insert(0, "..")

import torch
import torch_geometric.nn  # type: ignore
from tqdm import tqdm

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
    def __init__(self, feat_dim: int):
        super().__init__(feat_dim, feat_dim, torch.nn.functional.gelu)


class EdgeConvSubnet(Subnet):
    def __init__(self, feat_dim: int):
        super().__init__(2 * feat_dim, feat_dim, torch.nn.functional.elu)


class PostProcessing(torch.nn.Module):
    def __init__(self, feat_dim: int, output_dim: int):
        super().__init__()
        self.lin = torch_geometric.nn.Linear(feat_dim, output_dim)
        self.bn = torch.nn.BatchNorm1d(output_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, batch):
        h = x
        h = torch.nn.functional.elu(self.bn(self.lin(h)))
        h = torch_geometric.nn.pool.global_mean_pool(h, batch)
        return self.dropout(h)


class SharedPostProcessing(PostProcessing):
    def __init__(self, feat_dim: int):
        super().__init__(feat_dim, 2)


class SingleReceptorPostProcessing(PostProcessing):
    def __init__(self, feat_dim: int):
        super().__init__(feat_dim, 1)


class Model(torch.nn.Module):
    def __init__(
        self,
        num_middle_layers: int,
        feat_dim: int,
        num_initial_layers: int = 1,
        num_receptor_specific_layers: int = 0,
    ):
        super().__init__()
        self.shared_subblocks = torch.nn.ModuleList(
            [InitialSubnet(feat_dim) for _ in range(num_initial_layers)]
            + [
                torch_geometric.nn.EdgeConv(EdgeConvSubnet(feat_dim), aggr="max")
                for _ in range(num_middle_layers)
            ]
        )
        self.er_sublocks = torch.nn.ModuleList(
            [
                torch_geometric.nn.EdgeConv(EdgeConvSubnet(feat_dim), aggr="max")
                for _ in range(num_receptor_specific_layers)
            ]
        )
        self.pr_sublocks = torch.nn.ModuleList(
            [
                torch_geometric.nn.EdgeConv(EdgeConvSubnet(feat_dim), aggr="max")
                for _ in range(num_receptor_specific_layers)
            ]
        )
        self.shared_posts = torch.nn.ModuleList(
            [SharedPostProcessing(feat_dim) for _ in self.shared_subblocks]
        )
        self.er_posts = torch.nn.ModuleList(
            [SingleReceptorPostProcessing(feat_dim) for _ in self.er_sublocks]
        )
        self.pr_posts = torch.nn.ModuleList(
            [SingleReceptorPostProcessing(feat_dim) for _ in self.pr_sublocks]
        )

    def forward(self, x, edge_index, batch):
        block_outputs = []
        h = x

        for subblock, post in zip(self.shared_subblocks, self.shared_posts):
            if isinstance(subblock, torch_geometric.nn.MessagePassing):
                h = subblock(h, edge_index)
            else:
                h = subblock(h)
            block_outputs.append(post(h, batch))

        er_h = h
        pr_h = h
        for er_subblock, er_post, pr_subblock, pr_post in zip(
            self.er_sublocks, self.er_posts, self.pr_sublocks, self.pr_posts
        ):
            er_h = er_subblock(er_h, edge_index)
            er_out = er_post(er_h, batch)
            pr_h = pr_subblock(pr_h, edge_index)
            pr_out = pr_post(pr_h, batch)
            block_outputs.append(torch.concat((er_out, pr_out), dim=-1))

        agg: torch.Tensor = sum(block_outputs)  # type: ignore
        return agg


if __name__ == "__main__":
    trainer = GNNModelTrainer()
    for depth in tqdm([1, 2, 3, 4, 5, 6]):
        trainer.train_and_validate(partial(Model, depth - 1), f"econv_d{depth}")
