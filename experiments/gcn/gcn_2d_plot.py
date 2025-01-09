import glob
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

fig = plt.figure()

NUM_PLOTS = 2


def plot(x_data: NDArray, receptor_num: int, plot_num: int):
    axs = fig.add_subplot(
        2, NUM_PLOTS, receptor_num * NUM_PLOTS - (NUM_PLOTS - plot_num)
    )
    x, idxs = np.unique(x_data, return_index=True)
    y = np.empty_like(x, dtype=float)
    for i, j in enumerate(idxs):
        y[i] = np.mean(metrics[np.where(x_data == x_data[j])])
    axs.bar(x, y - y.min(), bottom=y.min())


paths = glob.glob("gcn_b*_w*.metrics")
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

    plot(depths, recepetor_num, 1)
    plot(decays, recepetor_num, 2)
plt.show()
