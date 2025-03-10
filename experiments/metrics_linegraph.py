# Used to create bar charts for presentation

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

models = [
    "Linear GNN",
    "GNN",
    "GCN 1 block only",
    "GCN",
    "GAT 1 block only",
    "GAT",
    "EdgeConv 1 block only",
    "EdgeConv",
]
model_param_counts = [
    2050,
    1053698,
    1053698,
    3156994,
    1055746,
    3163138,
    3155980,
    7360536,
]
er_metrics = ["AUCROC(ER)", "AUCROC(ER|PR−)"]
er_means = [
    [0.838],
    [0.853],
    [0.822],
    [0.863],
    [0.870],
    [0.871],
    [0.874],
    [0.880],
]
er_given_pr_means = [
    [0.780],
    [0.768],
    [0.746],
    [0.751],
    [0.800],
    [0.773],
    [0.818],
    [0.810],
]
pr_metrics = ["AUCROC(PR)", "AUCROC(PR|ER+)"]
pr_means = [
    [0.806],
    [0.786],
    [0.791],
    [0.812],
    [0.821],
    [0.828],
    [0.826],
    [0.833],
]
pr_given_er_means = [
    [0.674],
    [0.619],
    [0.675],
    [0.672],
    [0.700],
    [0.686],
    [0.684],
    [0.723],
]

for metric, means in zip(
    ("AUCROC(ER)", "AUCROC(ER|PR−)", "AUCROC(PR)", "AUCROC(PR|ER+)"),
    (er_means, er_given_pr_means, pr_means, pr_given_er_means),
):
    fig, ax = plt.subplots()
    for model, pc, mean in zip(models, model_param_counts, means):
        ax.scatter(
            pc * 1e-6,
            mean,
            label=model,
        )
    ax.legend()
    # ax.set_ylim(bottom=0.5, top=1)
    ax.set_xlabel("Trained parameters (millions)")
    ax.set_ylabel(metric)
    fig.tight_layout()
    plt.show()
