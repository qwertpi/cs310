from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12})

models = ["Linear", "GNN", "GCN", "GAT", "EdgeConv"]
er_metrics = ["AUCROC(ER)", "BALANCEDAUCROC(ER)", "AUCROC(ER|PR−)"]
er_means = np.array(
    [
        [0.838, 0.827, 0.753],
        [0.838, 0.819, 0.722],
        [0.845, 0.830, 0.741],
        [0.840, 0.825, 0.734],
        [0.885, 0.880, 0.828],
    ]
)
er_stds = np.array(
    [
        [0.044, 0.054, 0.099],
        [0.033, 0.044, 0.096],
        [0.044, 0.050, 0.099],
        [0.053, 0.056, 0.074],
        [0.034, 0.048, 0.113],
    ]
)
pr_metrics = ["AUCROC(PR)", "BALANCEDAUCROC(PR)", "AUCROC(PR|ER+)"]
pr_means = np.array(
    [
        [0.780, 0.812, 0.669],
        [0.780, 0.811, 0.670],
        [0.798, 0.829, 0.674],
        [0.801, 0.838, 0.668],
        [0.814, 0.850, 0.694],
    ]
)
pr_stds = np.array(
    [
        [0.050, 0.058, 0.047],
        [0.045, 0.052, 0.056],
        [0.035, 0.056, 0.033],
        [0.048, 0.062, 0.036],
        [0.053, 0.054, 0.063],
    ]
)

fig, ax = plt.subplots()
y_pos = 2 * len(models) * (len(er_metrics) + len(pr_metrics))
y_labels = {}
# color_pool = ["C0", "C1", "C9", "C4"]
color_pool = ["C0", "C1", "C9"]
colors = {
    m: color_pool[i % len(color_pool)] for i, m in enumerate(er_metrics + pr_metrics)
}
legend_data = {}


for means, stds, metric in chain(
    zip(er_means.T, er_stds.T, er_metrics), zip(pr_means.T, pr_stds.T, pr_metrics)
):
    for model, mean, std in zip(models, means, stds):
        y_labels[y_pos] = model
        bar = ax.barh(
            y_pos,
            mean,
            label=metric,
            color=colors[metric],
            xerr=std,
            capsize=5,
        )
        ax.text(0.51, bar[0].get_y() + 0.15, f"{mean:.3f}±{std:.3f}")
        legend_data[metric] = bar
        y_pos -= 1
    y_pos -= 1

ax.legend(legend_data.values(), legend_data.keys())
ax.set_yticks(list(y_labels.keys()), y_labels.values())
ax.set_xlim(left=0.5, right=1)
fig.tight_layout()
plt.show()
