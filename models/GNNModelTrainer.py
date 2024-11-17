from __future__ import annotations
from functools import partial
import pickle
from time import time
from typing import TYPE_CHECKING, Callable, ParamSpec

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

if torch.cuda.is_available():
    import cupy as np  # type: ignore
else:
    import numpy as np

P = ParamSpec("P")


def float_wrapper(f: Callable[P, ConvertibleToFloat]):
    def ret_func(*args: P.args, **kwargs: P.kwargs):
        return float(f(*args, **kwargs))

    return ret_func


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, graph, labels = self.data[idx]
        return graph, torch.tensor(labels, dtype=torch.float)


class GNNModelTrainer:
    def __init__(self):
        with open("../../../data/train_data.pkl", "rb") as f:
            data: list[tuple[str, Data, tuple[bool, bool]]] = pickle.load(f)
        self.dataset = Dataset(data)
        self.groups: list[str] = []
        self.dummy: list[None] = []
        self.y: list[tuple[bool, bool]] = []
        self.y_compact: list[str] = []
        for group, _, graph_label in data:
            self.y_compact.append(str(int(graph_label[0])) + str(int(graph_label[1])))
            self.y.append(graph_label)
            self.dummy.append(None)
            self.groups.append(group)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_and_validate(
        self,
        make_model: Callable[[], torch.nn.Module],
        model_name: str,
        batch_size: int,
        epochs: int,
    ):
        LABEL_METRICS: list[tuple[str, Callable[[list[int], list[int]], float]]] = [
            ("Sensitivity", recall_score),
            ("Specificity", partial(recall_score, pos_label=0)),
            ("Balanced accuracy", balanced_accuracy_score),
        ]
        PROBABILITY_METRICS: list[
            tuple[str, Callable[[list[int], list[float]], float]]
        ] = [
            ("AUC_ROC", float_wrapper(roc_auc_score)),
            ("AUC_PR", float_wrapper(average_precision_score)),
        ]
        ALL_METRICS = LABEL_METRICS + PROBABILITY_METRICS

        NUM_FOLDS = 5

        # [ER scores, PR scores]
        scores = np.empty((2, NUM_FOLDS, len(ALL_METRICS)))
        train_times = np.empty((NUM_FOLDS))
        folds = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True).split(
            self.dummy, self.y_compact, self.groups
        )
        # Delete the file if it already exists
        with open(f"{model_name}_ER.metrics", "w") as f:
            f.write("")
        with open(f"{model_name}_PR.metrics", "w") as f:
            f.write("")
        for fold_num, (train_idxs, validation_idxs) in enumerate(folds):
            with open(f"{model_name}_ER.metrics", "a") as f:
                f.write(f"{fold_num}\n")
            with open(f"{model_name}_PR.metrics", "a") as f:
                f.write(f"{fold_num}\n")

            train_loader = torch_geometric.loader.DataLoader(
                torch.utils.data.Subset(self.dataset, train_idxs),
                batch_size=batch_size,
                shuffle=True,
            )
            validation_loader = torch_geometric.loader.DataLoader(
                torch.utils.data.Subset(self.dataset, validation_idxs),
                batch_size=batch_size,
                shuffle=True,
            )

            t0 = time()
            model = make_model().to(self.device)
            optimizer = torch.optim.Adam(model.parameters())
            fig, ax = plt.subplots()
            er_losses = []
            pr_losses = []
            for epoch in range(epochs):
                er_loss = pr_loss = 0.0
                model.train()
                for batch_data, batch_y in train_loader:
                    batch_data, batch_y = batch_data.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    batch_er_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        out[:, 0], batch_y[:, 0]
                    )
                    batch_pr_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        out[:, 1], batch_y[:, 1]
                    )
                    loss = batch_er_loss + batch_pr_loss
                    loss.backward()
                    optimizer.step()
                model.eval()
                for batch_data, batch_y in validation_loader:
                    batch_data, batch_y = batch_data.to(self.device), batch_y.to(self.device)
                    out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    batch_er_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        out[:, 0], batch_y[:, 0]
                    )
                    batch_pr_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        out[:, 1], batch_y[:, 1]
                    )
                    er_loss += batch_er_loss.item()
                    pr_loss += batch_pr_loss.item()
                er_losses.append(er_loss)
                pr_losses.append(pr_loss)
            t1 = time()
            train_times[fold_num] = t1 - t0
            ax.plot(range(1, epochs + 1), er_losses)
            fig.savefig(f"{model_name}_loss_ER_{fold_num}.png")
            ax.clear()
            ax.plot(range(1, epochs + 1), pr_losses)
            fig.savefig(f"{model_name}_loss_PR_{fold_num}.png")
            with open(f"{model_name}_ER.metrics", "a") as f:
                f.write(f"Train time: {t1 - t0}\n")
            with open(f"{model_name}_PR.metrics", "a") as f:
                f.write(f"Train time: {t1 - t0}\n")

            y_pred = []
            y_true = []
            with torch.no_grad():
                for batch_data, batch_y in validation_loader:
                    batch_data, batch_y = batch_data.to(self.device), batch_y.to(self.device)
                    y_pred.extend(
                        model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    )
                    y_true.extend(batch_y)
            print(y_pred)
            print(y_true)
            for label_idx, label in enumerate(("ER", "PR")):
                metric_idx = 0
                y_pred_label = [max(0, min(1, round(t[label_idx].item()))) for t in y_pred]
                y_true_label = [max(0, min(1, round(t[label_idx].item()))) for t in y_true]
                for _, metric_func in LABEL_METRICS:
                    scores[label_idx, fold_num, metric_idx] = metric_func(
                        y_true_label, y_pred_label
                    )
                    metric_idx += 1
                y_pred_prob = [t[label_idx].item() for t in y_pred]
                y_true_prob = [t[label_idx].item() for t in y_true]
                for _, metric_func in PROBABILITY_METRICS:  # type: ignore
                    scores[label_idx, fold_num, metric_idx] = metric_func(
                        y_true_prob, y_pred_prob
                    )
                    metric_idx += 1

                with open(f"{model_name}_{label}.metrics", "a") as f:
                    for (metric_name, _), metric_val in zip(
                        ALL_METRICS, scores[label_idx, fold_num]
                    ):
                        f.write(f"{metric_name}: {metric_val}\n")

        for label_idx, label in enumerate(("ER", "PR")):
            means = np.mean(scores[label_idx], axis=0)
            std_devs = np.std(scores[label_idx], axis=0)

            with open(f"{model_name}_{label}.metrics", "a") as f:
                f.write("MEAN\n")
                for (metric_name, _), average in zip(ALL_METRICS, means):
                    f.write(f"{metric_name}: {average}\n")
                f.write(f"Train time: {np.mean(train_times)}\n")

                f.write("STD_DEV\n")
                for (metric_name, _), var in zip(ALL_METRICS, std_devs):
                    f.write(f"{metric_name}: {var}\n")
                f.write(f"Train time: {np.std(train_times)}\n")
