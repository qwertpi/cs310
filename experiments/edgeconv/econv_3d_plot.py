import glob
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
for recepetor_num, recepetor in ((1, "ER"), (2, "PR")):
    paths = glob.glob("econv_act_s*_m*.metrics")
    starts = np.empty(len(paths), dtype=int)
    middles = np.empty(len(paths), dtype=float)
    metrics = np.empty(len(paths), dtype=float)
    for i, path in enumerate(paths):
        start, tail = path.split("_s")[1].split("_m")
        starts[i] = start
        middles[i] = tail.split("_")[0].split(".metrics")[0]
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

    x, y = np.meshgrid(np.unique(starts), np.unique(middles))
    z = np.full_like(x, np.nan, dtype=float)
    for start, middle, metric in zip(starts, middles, metrics):
        z[np.where(np.logical_and(x == start, y == middle))] = metric
    ax = fig.add_subplot(1, 2, recepetor_num, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")  # type: ignore
plt.show()
