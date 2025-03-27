from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12})

models = ["Linear", "GNN", "GCN", "GAT", "EdgeConv"]
er_metrics = ["AUCROC(ER)", "BALANCEDAUCROC(ER)", "AUCROC(ER|PR−)"]
er_means = np.array(
    [
        [0.834, 0.819, 0.730],
        [0.850, 0.836, 0.752],
        [0.855, 0.842, 0.764],
        [0.845, 0.831, 0.748],
        [0.893, 0.886, 0.824],
    ]
)
er_stds = np.array(
    [
        [0.040, 0.050, 0.099],
        [0.034, 0.043, 0.077],
        [0.057, 0.071, 0.133],
        [0.048, 0.045, 0.078],
        [0.027, 0.026, 0.078],
    ]
)
pr_metrics = ["AUCROC(PR)", "BALANCEDAUCROC(PR)", "AUCROC(PR|ER+)"]
pr_means = np.array(
    [
        [0.782, 0.814, 0.670],
        [0.784, 0.819, 0.664],
        [0.799, 0.835, 0.663],
        [0.789, 0.826, 0.660],
        [0.814, 0.851, 0.679],
    ]
)
pr_stds = np.array(
    [
        [0.042, 0.052, 0.036],
        [0.042, 0.053, 0.033],
        [0.054, 0.065, 0.065],
        [0.041, 0.049, 0.049],
        [0.053, 0.055, 0.071],
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
