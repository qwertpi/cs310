import glob
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12})

er_weights = []
pr_weights = []
metrics: dict[str, list[float]] = {
    "AUC_ROC_ER": [],
    "AUC_ROC_ER_BALANCED": [],
    "AUC_ROC_ER|PR-": [],
    "AUC_ROC_PR": [],
    "AUC_ROC_PR_BALANCED": [],
    "AUC_ROC_PR|ER+": [],
}
for path in glob.glob("econv_sd*.metrics"):
    er_weight, tail = path.split("_e")[1].split("_p")
    er_weights.append(float(er_weight))
    pr_weight = tail.split("_p")[0].split(".metrics")[0]
    pr_weights.append(float(pr_weight))
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
                metrics[metric].append(float(split_line[1]))

fig = plt.figure()
means = [
    (m1 + m2) / 2
    for m1, m2 in zip(metrics["AUC_ROC_ER|PR-"], metrics["AUC_ROC_PR|ER+"])
]
# for i, (metric_name, means) in enumerate(metrics.items(), start=1):
x, y = np.meshgrid(np.unique(er_weights), np.unique(pr_weights))
z = np.full_like(x, np.nan, dtype=float)
for er_w, pr_w, mean in zip(er_weights, pr_weights, means):
    z[np.where(np.logical_and(x == er_w, y == pr_w))] = mean
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.plot_surface(x, y, z, cmap="viridis")  # type: ignore
ax.set_xlabel("log_10(er_weight)")
ax.set_ylabel("log_10(pr_weight)")
ax.set_title("((AUC_ROC_ER|PR-)+(AUC_ROC_PR|ER+))/2")
plt.show()
