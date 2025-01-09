import glob
import numpy as np
from matplotlib import pyplot as plt

paths = glob.glob("gcn_b*_w*.metrics")
fig = plt.figure()
for recepetor_num, receptor_paths in enumerate(
    (
        (path for path in paths if "_ER." in path),
        (path for path in paths if "_PR." in path),
    ),
    start=1,
):
    depths = np.empty(len(paths) // 2, dtype=int)
    decays = np.empty(len(paths) // 2, dtype=float)
    metrics = np.empty(len(paths) // 2, dtype=float)
    for i, path in enumerate(receptor_paths):
        depth, tail = path.split("_b")[1].split("_w")
        depths[i] = depth
        decays[i] = tail.split("_")[0]
        decays[i] = np.log10(decays[i]) if decays[i] != 0 else 0
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

    x, y = np.meshgrid(np.unique(depths), np.unique(decays))
    z = np.full_like(x, np.nan, dtype=float)
    for depth, decay, metric in zip(depths, decays, metrics):
        z[np.where(np.logical_and(x == depth, y == decay))] = metric
    ax = fig.add_subplot(1, 2, recepetor_num, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")  # type: ignore
plt.show()
