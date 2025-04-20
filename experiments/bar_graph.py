import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 26})

ys = ["Before", "After"]
xs = ["ER", "ER (PR Balanced)", "ER|PR–", "PR", "PR (ER Balanced)", "PR|ER+"]

# fmt: off
means = np.array(
    [
        [0.893, 0.886, 0.824, 0.814, 0.851, 0.679,],
        [0.871, 0.855, 0.760, 0.824, 0.867, 0.676,],
    ]
)
stds = np.array(
    [
        [0.027, 0.026, 0.078, 0.053, 0.055, 0.071,],
        [0.030, 0.031, 0.072, 0.045, 0.044, 0.079,],
    ]
)
# fmt: on

fig, ax = plt.subplots()
y_pos = 2 * len(ys) * len(xs)
y_labels = {}
color_pool = ["C0", "C6", "C8"]
colors = {m: color_pool[i % len(color_pool)] for i, m in enumerate(xs)}
legend_data = {}


for x, x_means, x_stds in zip(xs, means.T, stds.T):
    for y, mean, std in zip(ys, x_means, x_stds):
        y_labels[y_pos] = y
        bar = ax.barh(
            y_pos,
            mean,
            label=x,
            color=colors[x],
            xerr=std,
            capsize=5,
        )
        ax.text(0.51, bar[0].get_y() + 0.1, f"{mean:.3f}±{std:.3f}")
        legend_data[x] = bar
        y_pos -= 1
    y_pos -= 1

ax.legend(legend_data.values(), legend_data.keys())
ax.set_yticks(list(y_labels.keys()), y_labels.values())
ax.set_xlim(left=0.5, right=1)
fig.tight_layout()
plt.show()
