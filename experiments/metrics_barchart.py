# Used to create bar charts for presentation

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

models = ["Linear GNN", "GNN", "GCN", "GAT", "EdgeConv"]
er_metrics = ["AUCROC(ER)", "AUCROC(ER|PR−)"]
er_means = [
    [0.838, 0.780],
    [0.837, 0.753],
    [0.863, 0.751],
    [0.871, 0.773],
    [0.880, 0.810],
]
er_stds = [
    [0.042, 0.062],
    [0.050, 0.062],
    [0.019, 0.065],
    [0.037, 0.054],
    [0.023, 0.073],
]
pr_metrics = ["AUCROC(PR)", "AUCROC(PR|ER+)"]
pr_means = [
    [0.806, 0.674],
    [0.799, 0.692],
    [0.812, 0.672],
    [0.828, 0.686],
    [0.833, 0.723],
]
pr_stds = [
    [0.038, 0.074],
    [0.043, 0.112],
    [0.026, 0.053],
    [0.046, 0.108],
    [0.059, 0.105],
]

fig, ax = plt.subplots()
y_pos = 2 * len(models) * (len(er_metrics) + len(pr_metrics))
y_labels = {}
color_pool = ["C0", "C1", "C9", "C4"]
colors = {m: color_pool[i] for i, m in enumerate(er_metrics + pr_metrics)}
legend_data = {}

for model, means, stds in zip(models, er_means, er_stds):
    y_labels[y_pos] = model
    for mean, std, metric in zip(means, stds, er_metrics):
        bar = ax.barh(
            y_pos,
            mean,
            height=1,
            label=metric,
            color=colors[metric],
            xerr=std,
            capsize=5,
        )
        ax.text(0.51, bar[0].get_y()+0.25, f"{mean:.3f}±{std:.3f}")
        legend_data[metric] = bar
        y_pos -= 1
    y_pos -= 1

y_pos -= 1

for model, means, stds in zip(models, pr_means, pr_stds):
    y_labels[y_pos] = model
    for mean, std, metric in zip(means, stds, pr_metrics):
        bar = ax.barh(
            y_pos,
            mean,
            height=1,
            label=metric,
            color=colors[metric],
            xerr=std,
            capsize=5,
        )
        ax.text(0.51, bar[0].get_y()+0.25, f"{mean:.3f}±{std:.3f}")
        legend_data[metric] = bar
        y_pos -= 1
    y_pos -= 1

ax.legend(legend_data.values(), legend_data.keys())
ax.set_yticks(list(y_labels.keys()), y_labels.values())
ax.set_xlim(left=0.5, right=1)
fig.tight_layout()
plt.show()
