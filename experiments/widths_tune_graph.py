from collections import defaultdict
import glob
from typing import cast
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 26})

means: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
stds: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)

for path in glob.glob("*/*_w*.metrics"):
    model_name, tail = path.split("/")[1].split("_")
    width = int(tail.split("w")[1].split(".metrics")[0])
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
            if averages:
                if metric == "AUC_ROC_ER":
                    er_mean = float(split_line[1])
                elif metric == "AUC_ROC_PR":
                    er_mean = cast(float, er_mean)  # type: ignore
                    pr_mean = float(split_line[1])
                    means[model_name][width] = (er_mean, pr_mean)
            if std_devs:
                if metric == "AUC_ROC_ER":
                    er_std = float(split_line[1])
                elif metric == "AUC_ROC_PR":
                    er_std = cast(float, er_std)  # type: ignore
                    pr_std = float(split_line[1])
                    stds[model_name][width] = (er_std, pr_std)

colors = {"AUC_ROC(ER)": "C3", "AUC_ROC(PR)": "C4"}
for i, metric in enumerate(("AUC_ROC(ER)", "AUC_ROC(PR)")):
    fig = plt.figure()
    for j, model in enumerate(("gnn", "gcn", "gat", "econv"), start=1):
        ax = fig.add_subplot(2, 2, j)
        model_means = means[model]
        model_stds = stds[model]
        widths, widths_means, widths_stds = zip(
            *(
                (
                    (w, m, s)
                    for w, m, s in sorted(
                        ((w, m[i], model_stds[w][i]) for w, m in model_means.items()),
                        key=lambda t: t[0],
                    )
                )
            )
        )
        ax.errorbar(
            widths,
            widths_means,
            yerr=widths_stds,
            capsize=5,
            color=colors[metric],
            marker="x",
        )
        ax.set_title(
            {"gnn": "GNN", "gcn": "GCN", "gat": "GAT", "econv": "EdgeConv"}[model]
        )
        ax.set_xlabel("log2(hidden_dimension)")
        ax.set_xlim(left=-0.25, right=10.25)
        if metric == "AUC_ROC(ER)":
            ax.set_ylim(bottom=0.70, top=0.95)
        elif metric == "AUC_ROC(PR)":
            ax.set_ylim(bottom=0.70, top=0.9)
        else:
            raise RuntimeError
        ax.set_ylabel(metric)
    fig.tight_layout()
    plt.show()
