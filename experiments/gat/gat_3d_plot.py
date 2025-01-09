import glob
import numpy as np
from matplotlib import pyplot as plt

paths = glob.glob("gat_h*_b*.metrics")
fig = plt.figure()
for plot_num, receptor_paths in enumerate(
    (
        (path for path in paths if "_ER." in path),
        (path for path in paths if "_PR." in path),
    ),
    start=1,
):
    widths = np.empty(len(paths) // 2, dtype=int)
    depths = np.empty(len(paths) // 2, dtype=int)
    metrics = np.empty(len(paths) // 2, dtype=float)
    for i, path in enumerate(receptor_paths):
        width, tail = path.split("_h")[1].split("_b")
        depth = tail.split("_")[0]
        widths[i] = width
        depths[i] = depth
        with open(path, "r") as f:
            averages = False
            for line in f.read().splitlines():
                if line == "MEAN":
                    averages = True
                    continue
                if not averages:
                    continue

                split_line = line.split(": ")
                if split_line[0] == "AUC_ROC":
                    metrics[i] = float(split_line[1])
                    break

    x, y = np.meshgrid(np.unique(widths), np.unique(depths))
    z = np.full_like(x, np.nan, dtype=float)
    for width, depth, metric in zip(widths, depths, metrics):
        z[np.where(np.logical_and(x == width, y == depth))] = metric
    ax = fig.add_subplot(1, 2, plot_num, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")  # type: ignore
plt.show()
