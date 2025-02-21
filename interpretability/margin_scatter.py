from abc import ABC, abstractmethod
import pickle
from typing import BinaryIO, Callable

import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import torch_geometric.nn  # type: ignore
from tqdm import tqdm


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


@click.command()
@click.argument("model_file", type=click.File("rb"))
@click.argument("data_file", type=click.File("rb"))
def pr_margin_scatter(model_file: BinaryIO, data_file: BinaryIO):
    model = Model(1, 3)
    state_dict = {
        k.replace("model.", ""): v
        for k, v in torch.load(model_file)["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    data = pickle.load(data_file)
    graphs = [t[1] for t in data]
    true_labels = np.array([t[2] for t in data])

    model.eval()
    with torch.no_grad():
        predictions = np.array(
            [
                torch.nn.functional.sigmoid(
                    model(
                        graph.x,
                        graph.edge_index,
                        torch.zeros(graph.x.size(0), dtype=torch.long),
                    )[0][1]
                )
                for graph in tqdm(graphs)
            ]
        )

    sort_idxs = np.argsort(predictions)
    predictions = predictions[sort_idxs]
    true_labels = true_labels[sort_idxs]

    y = predictions
    x = np.empty_like(y)
    GAP = 0.05
    last_y: dict[float, float] = {}
    for i in range(len(y)):
        candidate_x = 0.0
        while last_y.get(candidate_x) and abs(y[i] - last_y[candidate_x]) < GAP:
            candidate_x += GAP
        x[i] = candidate_x
        last_y[candidate_x] = y[i]

    plt.figure()
    er_pos_pr_pos_mask = (true_labels[:, 0] == True) & (true_labels[:, 1] == True)
    plt.scatter(
        x[er_pos_pr_pos_mask],
        y[er_pos_pr_pos_mask],
        marker="o",
        color="green",
        label="ER+PR+",
    )
    er_neg_pr_pos_mask = (true_labels[:, 0] == False) & (true_labels[:, 1] == True)
    plt.scatter(
        x[er_neg_pr_pos_mask],
        y[er_neg_pr_pos_mask],
        marker="s",
        color="green",
        label="ER-PR+",
    )
    er_pos_pr_neg_mask = (true_labels[:, 0] == True) & (true_labels[:, 1] == False)
    plt.scatter(
        x[er_pos_pr_neg_mask],
        y[er_pos_pr_neg_mask],
        marker="o",
        color="red",
        label="ER+PR-",
    )
    er_neg_pr_neg_mask = (true_labels[:, 0] == False) & (true_labels[:, 1] == False)
    plt.scatter(
        x[er_neg_pr_neg_mask],
        y[er_neg_pr_neg_mask],
        marker="s",
        color="red",
        label="ER-PR+",
    )
    plt.xticks([])
    plt.legend()
    plt.savefig("scatter_all.png")
    print(roc_auc_score(true_labels[:, 1], predictions))

    plt.clf()
    er_pos_pr_pos_mask = (true_labels[:, 0] == True) & (true_labels[:, 1] == True)
    plt.scatter(
        x[er_pos_pr_pos_mask],
        y[er_pos_pr_pos_mask],
        marker="o",
        color="green",
    )
    er_pos_pr_neg_mask = (true_labels[:, 0] == True) & (true_labels[:, 1] == False)
    plt.scatter(
        x[er_pos_pr_neg_mask],
        y[er_pos_pr_neg_mask],
        marker="o",
        color="red",
    )
    plt.xticks([])
    plt.savefig("scatter_ER+.png")
    print(roc_auc_score(true_labels[:, 1][true_labels[:, 0] == True], predictions[true_labels[:, 0] == True]))

    plt.clf()
    er_neg_pr_pos_mask = (true_labels[:, 0] == False) & (true_labels[:, 1] == True)
    plt.scatter(
        x[er_neg_pr_pos_mask],
        y[er_neg_pr_pos_mask],
        marker="s",
        color="green",
        label="ER-PR+",
    )
    er_neg_pr_neg_mask = (true_labels[:, 0] == False) & (true_labels[:, 1] == False)
    plt.scatter(
        x[er_neg_pr_neg_mask],
        y[er_neg_pr_neg_mask],
        marker="s",
        color="red",
        label="ER-PR-",
    )
    plt.xticks([])
    plt.savefig("scatter_ER-.png")
    print(roc_auc_score(true_labels[:, 1][true_labels[:, 0] == False], predictions[true_labels[:, 0] == False]))

    with torch.no_grad():
        predictions = np.array(
            [
                torch.nn.functional.sigmoid(
                    model(
                        graph.x,
                        graph.edge_index,
                        torch.zeros(graph.x.size(0), dtype=torch.long),
                    )[0][0]
                )
                for graph in tqdm(graphs)
            ]
        )

    sort_idxs = np.argsort(predictions)
    predictions = predictions[sort_idxs]
    true_labels = true_labels[sort_idxs]

    y = predictions
    x = np.empty_like(y)
    GAP = 0.05
    last_y: dict[float, float] = {}
    for i in range(len(y)):
        candidate_x = 0.0
        while last_y.get(candidate_x) and abs(y[i] - last_y[candidate_x]) < GAP:
            candidate_x += GAP
        x[i] = candidate_x
        last_y[candidate_x] = y[i]

    plt.figure()
    er_pos_pr_pos_mask = (true_labels[:, 0] == True) & (true_labels[:, 1] == True)
    plt.scatter(
        x[er_pos_pr_pos_mask],
        y[er_pos_pr_pos_mask],
        marker="o",
        color="green",
        label="ER+PR+",
    )
    er_neg_pr_pos_mask = (true_labels[:, 0] == False) & (true_labels[:, 1] == True)
    plt.scatter(
        x[er_neg_pr_pos_mask],
        y[er_neg_pr_pos_mask],
        marker="o",
        color="red",
        label="ER-PR+",
    )
    er_pos_pr_neg_mask = (true_labels[:, 0] == True) & (true_labels[:, 1] == False)
    plt.scatter(
        x[er_pos_pr_neg_mask],
        y[er_pos_pr_neg_mask],
        marker="s",
        color="green",
        label="ER+PR-",
    )
    er_neg_pr_neg_mask = (true_labels[:, 0] == False) & (true_labels[:, 1] == False)
    plt.scatter(
        x[er_neg_pr_neg_mask],
        y[er_neg_pr_neg_mask],
        marker="s",
        color="red",
        label="ER-PR-",
    )
    plt.xticks([])
    plt.legend()
    plt.savefig("scatter_er_all.png")
    print(roc_auc_score(true_labels[:, 0], predictions))

    plt.clf()
    er_pos_pr_pos_mask = (true_labels[:, 0] == True) & (true_labels[:, 1] == True)
    plt.scatter(
        x[er_pos_pr_pos_mask],
        y[er_pos_pr_pos_mask],
        marker="o",
        color="green",
        label="ER+PR+",
    )
    er_neg_pr_pos_mask = (true_labels[:, 0] == False) & (true_labels[:, 1] == True)
    plt.scatter(
        x[er_neg_pr_pos_mask],
        y[er_neg_pr_pos_mask],
        marker="o",
        color="red",
        label="ER-PR+",
    )
    plt.savefig("scatter_PR+.png")
    print(roc_auc_score(true_labels[:, 0][true_labels[:, 1] == True], predictions[true_labels[:, 1] == True]))

    plt.clf()
    er_pos_pr_neg_mask = (true_labels[:, 0] == True) & (true_labels[:, 1] == False)
    plt.scatter(
        x[er_pos_pr_neg_mask],
        y[er_pos_pr_neg_mask],
        marker="s",
        color="green",
        label="ER+PR-",
    )
    er_neg_pr_neg_mask = (true_labels[:, 0] == False) & (true_labels[:, 1] == False)
    plt.scatter(
        x[er_neg_pr_neg_mask],
        y[er_neg_pr_neg_mask],
        marker="s",
        color="red",
        label="ER-PR-",
    )
    plt.savefig("scatter_PR-.png")
    print(roc_auc_score(true_labels[:, 0][true_labels[:, 1] == False], predictions[true_labels[:, 1] == False]))


if __name__ == "__main__":
    pr_margin_scatter()
