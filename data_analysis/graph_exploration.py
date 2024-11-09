import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from networkx import connected_components, diameter  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from torch import Tensor
import torch_geometric.data  # type: ignore
import torch_geometric.utils  # type: ignore
from tqdm import tqdm

with open("../../data/train_data.pkl", "rb") as f:
    train_set: list[tuple[str, torch_geometric.data.Data, tuple[bool, bool]]] = (
        pickle.load(f)
    )
with open("../../data/test_data.pkl", "rb") as f:
    test_set: list[tuple[str, torch_geometric.data.Data, tuple[bool, bool]]] = (
        pickle.load(f)
    )

for dataset, dataset_name in ((test_set, "test"), (train_set, "train")):
    data_list: list[Tensor] = []
    diameter_list: list[int] = []

    _: Any
    for _, graph, _ in tqdm(dataset):
        node_features = graph.x
        if node_features is None:
            print("No features")
        elif np.isnan(node_features.numpy()).any():
            print("NaN found: ", node_features)
        else:
            nx_g = torch_geometric.utils.to_networkx(graph).to_undirected()
            diameter_list.append(
                max((diameter(nx_g.subgraph(cc)) for cc in connected_components(nx_g)))
            )
            data_list += node_features

    plt.clf()
    plt.hist(diameter_list)
    plt.savefig(f"{dataset_name}_diameters.png")

    data = make_pipeline(PCA(n_components=8)).fit_transform(data_list)
    for i, feats in enumerate(data.T):
        plt.clf()
        plt.hist(feats)
        plt.savefig(f"{dataset_name}_feature_{i}.png")
