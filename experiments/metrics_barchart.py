from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12})

models = ["Linear", "GNN", "GCN", "GAT", "EdgeConv"]
er_metrics = ["AUCROC(ER)", "BALANCEDAUCROC(ER)", "AUCROC(ER|PR−)"]
er_means = np.array(
    [
        [0.890, 0.880, 0.820],
        [0.917, 0.906, 0.845],
        [0.913, 0.904, 0.852],
        [0.906, 0.896, 0.835],
        [0.922, 0.911, 0.848],
    ]
)
er_stds = np.array(
    [
        [0.020, 0.020, 0.032],
        [0.016, 0.018, 0.027],
        [0.022, 0.022, 0.041],
        [0.016, 0.019, 0.037],
        [0.010, 0.011, 0.036],
    ]
)
pr_metrics = ["AUCROC(PR)", "BALANCEDAUCROC(PR)", "AUCROC(PR|ER+)"]
pr_means = np.array(
    [
        [0.796, 0.841, 0.599],
        [0.810, 0.846, 0.650],
        [0.795, 0.835, 0.623],
        [0.805, 0.845, 0.621],
        [0.821, 0.858, 0.645],
    ]
)
pr_stds = np.array(
    [
        [0.014, 0.019, 0.051],
        [0.011, 0.019, 0.034],
        [0.024, 0.022, 0.051],
        [0.011, 0.020, 0.047],
        [0.012, 0.013, 0.051],
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
