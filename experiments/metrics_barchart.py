from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12})

models = ["Linear", "GNN", "GCN", "GAT", "EdgeConv"]
er_metrics = ["AUCROC(ER)", "BALANCEDAUCROC(ER)", "AUCROC(ER|PR−)"]
er_means = np.array(
    [
        [0.715, 0.718, 0.708],
        [0.692, 0.692, 0.675],
        [0.764, 0.763, 0.736],
        [0.718, 0.717, 0.692],
        [0.812, 0.819, 0.814],
    ]
)
er_stds = np.array(
    [
        [0.057, 0.073, 0.144],
        [0.032, 0.055, 0.145],
        [0.041, 0.059, 0.136],
        [0.027, 0.047, 0.120],
        [0.029, 0.044, 0.101],
    ]
)
pr_metrics = ["AUCROC(PR)", "BALANCEDAUCROC(PR)", "AUCROC(PR|ER+)"]
pr_means = np.array(
    [
        [0.780, 0.804, 0.680],
        [0.723, 0.740, 0.643],
        [0.783, 0.805, 0.698],
        [0.762, 0.782, 0.673],
        [0.806, 0.849, 0.662],
    ]
)
pr_stds = np.array(
    [
        [0.045, 0.056, 0.084],
        [0.068, 0.070, 0.099],
        [0.036, 0.028, 0.072],
        [0.055, 0.057, 0.088],
        [0.049, 0.031, 0.113],
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
