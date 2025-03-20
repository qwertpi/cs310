"""
This entire file is adapated from: https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/full-pipelines/slide-graph.html#visualize-a-sample-graph
Date accessed: 2024-10-23
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
from skimage.exposure import equalize_hist  # type: ignore
from tiatoolbox.wsicore.wsireader import WSIReader  # type: ignore
from tiatoolbox.utils.visualization import plot_graph  # type: ignore

for wsi_name in os.listdir("../../data/wsis/"):
    barcode = "-".join(wsi_name.split("-")[0:3])
    with open(f"../../data/graphs/{barcode}_ShuffleNet.pkl", "rb") as f:
        g = pickle.load(f)

    # Project features down into 3D to be used as R G B colour channels
    pca = PCA(n_components=3).fit_transform(g.x)
    colours = np.empty(pca.shape)
    for channel in range(3):
        colours[:, channel] = (1 - equalize_hist(pca[:, channel]) ** 2) * 255

    wsi_reader = WSIReader.open(f"../../data/wsis/{wsi_name}")
    thumb = wsi_reader.slide_thumbnail(4.0, "mpp")
    coordinates = (
        np.array(g.coords)
        * np.array(wsi_reader.slide_dimensions(4, "mpp"))
        / np.array(wsi_reader.slide_dimensions(0.5, "mpp"))
    )

    thumb_and_graph = plot_graph(
        thumb.copy(), coordinates, g.edge_index.T, node_colors=colours, node_size=24
    )
    plt.subplot(1, 2, 1)
    plt.imshow(thumb)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(thumb_and_graph)
    plt.axis("off")
    plt.savefig(f"{barcode}.png", dpi=1200)
