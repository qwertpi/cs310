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

    x, idxs = np.unique(np.log2(widths), return_index=True)
    y = np.empty_like(x, dtype=float)
    for i in idxs:
        y[i] = np.mean(metrics[np.where(widths == widths[i])])
    ax1 = fig.add_subplot(2, 2, plot_num * 2 - 1)
    ax1.bar(x, y - y.min(), bottom=y.min())
    x, idxs = np.unique(depths, return_index=True)
    print(x)
    y = np.empty_like(x, dtype=float)
    for i in idxs:
        y[i] = np.mean(metrics[np.where(depths == depths[i])])
    print(y)
    ax2 = fig.add_subplot(2, 2, plot_num * 2)
    ax2.bar(x, y - y.min(), bottom=y.min())
plt.show()
