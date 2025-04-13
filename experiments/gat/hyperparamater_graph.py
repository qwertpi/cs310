from math import log2
import glob
from typing import cast

from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 26})

heads: list[int] = []
metrics: dict[str, tuple[list[float], list[float]]] = {
    "AUC_ROC_ER": ([], []),
    "AUC_ROC_PR": ([], []),
}

for path in glob.glob("gat_h*.metrics"):
    head, tail = path.split("_h")[1].split(".metrics")
    heads.append(int(log2(int(head))))
    with open(path, "r") as f:
        averages = False
        std_devs = False
        for line in f.read().splitlines():
            if line == "MEAN":
                averages = True
                continue
            if line == "STD_DEV":
                averages = False
                std_devs = True
                continue
            if not averages and not std_devs:
                continue
            split_line = line.split(": ")
            metric = split_line[0]
            if averages and metric in metrics.keys():
                metrics[metric][0].append(float(split_line[1]))
            if std_devs and metric in metrics.keys():
                metrics[metric][1].append(float(split_line[1]))

colors = {"AUC_ROC_ER": "C3", "AUC_ROC_PR": "C4"}
fig = plt.figure()
for i, (metric, (means, stds)) in enumerate(metrics.items(), start=1):
    heads, means, stds = cast(
        tuple[list[int], list[float], list[float]],
        zip(*sorted(zip(heads, means, stds), key=lambda t: t[0])),
    )
    ax = fig.add_subplot(2, 1, i)
    ax.errorbar(
        heads,
        means,
        label=metric,
        color=colors[metric],
        yerr=stds,
        capsize=5,
        marker="x",
    )
    ax.set_xlabel("log2(head_count)")
    if metric == "AUC_ROC_ER":
        ax.set_ylim(bottom=0.75, top=0.95)
        ax.set_ylabel("AUC_ROC(ER)")
    elif metric == "AUC_ROC_PR":
        ax.set_ylim(bottom=0.7, top=0.85)
        ax.set_ylabel("AUC_ROC(PR)")
    else:
        raise RuntimeError
fig.tight_layout()
plt.show()
