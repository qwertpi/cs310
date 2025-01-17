import glob
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

fig = plt.figure()

NUM_PLOTS = 1


def plot(x_data: NDArray, receptor_num: int, plot_num: int):
    axs = fig.add_subplot(
        2, NUM_PLOTS, receptor_num * NUM_PLOTS - (NUM_PLOTS - plot_num)
    )
    x, idxs = np.unique(x_data, return_index=True)
    y = np.empty_like(x, dtype=float)
    for i, j in enumerate(idxs):
        y[i] = np.mean(metrics[np.where(x_data == x_data[j])])
    axs.bar(x, y - y.min(), bottom=y.min(), width=0.2)


paths = glob.glob("logreg_c=*.metrics")
for receptor, recepetor_num in (("ER", 1), ("PR", 2)):
    cs = np.empty(len(paths), dtype=float)
    metrics = np.empty(len(paths), dtype=float)
    for i, path in enumerate(paths):
        start, tail = path.split("_c=")[1].split(".metrics")
        cs[i] = np.log10(float(start))
        with open(path, "r") as f:
            averages = False
            for line in f.read().splitlines():
                if line == "MEAN":
                    averages = True
                    continue
                if not averages:
                    continue

                split_line = line.split(": ")
                if split_line[0] == f"AUC_ROC_{receptor}":
                    metrics[i] = float(split_line[1])
                    break
    plot(cs, recepetor_num, 1)
plt.show()
