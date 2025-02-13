import glob
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
for recepetor_num, recepetor in ((1, "ER"), (2, "PR")):
    paths = glob.glob("gnn_a*_d*.metrics")
    activations = np.empty(len(paths), dtype=int)
    decays = np.empty(len(paths), dtype=float)
    metrics = np.empty(len(paths), dtype=float)
    for i, path in enumerate(paths):
        start, tail = path.split("_a")[1].split("_d")
        activations[i] = start
        start, tail = tail.split(".metrics")
        decays[i] = start
        with open(path, "r") as f:
            averages = False
            for line in f.read().splitlines():
                if line == "MEAN":
                    averages = True
                    continue
                if not averages:
                    continue

                split_line = line.split(": ")
                if split_line[0] == f"AUC_ROC_{recepetor}":
                    metrics[i] = float(split_line[1])
                    break

    decays = np.log10(decays)
    x, y = np.meshgrid(np.unique(activations), np.unique(decays))
    z = np.full_like(x, np.nan, dtype=float)
    for activation, decay, metric in zip(activations, decays, metrics):
        z[np.where(np.logical_and(x == activation, y == decay))] = metric
    ax = fig.add_subplot(1, 2, recepetor_num, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")  # type: ignore
plt.show()
