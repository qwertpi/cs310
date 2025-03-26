from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12})

models = ["Linear", "GNN", "GCN", "GAT", "EdgeConv"]
er_metrics = ["AUCROC(ER)", "BALANCEDAUCROC(ER)", "AUCROC(ER|PR−)"]
er_means = np.array(
    [
        [0.888, 0.878, 0.826],
        [0.863, 0.854, 0.820],
        [0.850, 0.842, 0.799],
        [0.871, 0.863, 0.823],
        [0.892, 0.884, 0.847],
    ]
)
er_stds = np.array(
    [
        [0.012, 0.012, 0.028],
        [0.037, 0.041, 0.053],
        [0.018, 0.020, 0.034],
        [0.012, 0.017, 0.036],
        [0.013, 0.017, 0.036],
    ]
)
pr_metrics = ["AUCROC(PR)", "BALANCEDAUCROC(PR)", "AUCROC(PR|ER+)"]
pr_means = np.array(
    [
        [0.811, 0.852, 0.624],
        [0.777, 0.815, 0.598],
        [0.808, 0.853, 0.625],
        [0.794, 0.840, 0.603],
        [0.808, 0.849, 0.614],
    ]
)
pr_stds = np.array(
    [
        [0.014, 0.018, 0.054],
        [0.020, 0.033, 0.039],
        [0.006, 0.015, 0.040],
        [0.018, 0.018, 0.053],
        [0.011, 0.022, 0.063],
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
