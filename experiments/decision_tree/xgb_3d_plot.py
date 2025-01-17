import glob
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
for recepetor_num, recepetor in ((1, "ER"), (2, "PR")):
    paths = glob.glob("xgb_g*_l1*_l2*.metrics")
    l1s = np.empty(len(paths), dtype=float)
    l2s = np.empty(len(paths), dtype=float)
    metrics = np.empty(len(paths), dtype=float)
    for i, path in enumerate(paths):
        start, tail = path.split("_l1")[1].split("_l2")
        l1s[i] = np.log10(float(start)) if start != "0" else -1
        start, tail = tail.split(".metrics")
        l2s[i] = np.log10(float(start)) if start != "0" else -1
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

    x, y = np.meshgrid(np.unique(l1s), np.unique(l2s))
    z = np.full_like(x, np.nan, dtype=float)
    for l1, l2, metric in zip(l1s, l2s, metrics):
        z[np.where(np.logical_and(x == l1, y == l2))] = metric
    ax = fig.add_subplot(1, 2, recepetor_num, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")  # type: ignore
plt.show()
