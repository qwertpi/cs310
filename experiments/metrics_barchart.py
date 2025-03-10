# Used to create bar charts for presentation

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

models = [
    "3 Shared 0 Separate",
    "2 Shared 1 Separate",
    "1 Shared 2 Separate",
    "0 Shared 3 Separate",
]
er_metrics = ["AUCROC(ER)", "AUCROC(ER|PR−)"]
er_means = [
    [0.900, 0.839],
    [0.891, 0.819],
    [0.878, 0.801],
    [0.851, 0.783],
]
er_stds = [
    [0.026, 0.071],
    [0.041, 0.094],
    [0.028, 0.097],
    [0.100, 0.103],
]
pr_metrics = ["AUCROC(PR)", "AUCROC(PR|ER+)"]
pr_means = [
    [0.834, 0.687],
    [0.835, 0.690],
    [0.822, 0.658],
    [0.795, 0.625],
]
pr_stds = [
    [0.031, 0.068],
    [0.050, 0.101],
    [0.040, 0.105],
    [0.081, 0.106],
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
        ax.text(0.51, bar[0].get_y() + 0.25, f"{mean:.3f}±{std:.3f}")
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
        ax.text(0.51, bar[0].get_y() + 0.25, f"{mean:.3f}±{std:.3f}")
        legend_data[metric] = bar
        y_pos -= 1
    y_pos -= 1

ax.legend(legend_data.values(), legend_data.keys())
ax.set_yticks(list(y_labels.keys()), y_labels.values())
ax.set_xlim(left=0.5, right=1)
fig.tight_layout()
plt.show()
