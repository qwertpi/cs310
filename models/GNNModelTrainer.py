from __future__ import annotations
from functools import partial
import pickle
from time import time
from typing import TYPE_CHECKING, Callable, ParamSpec

from pytorch_lightning.loggers import CSVLogger

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
import torch

from torch_geometric.data import Data  # type: ignore
import torch_geometric.loader  # type: ignore
import torch_geometric.nn  # type: ignore

torch.set_float32_matmul_precision("medium")

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
        graph.y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)
        return graph


class LightningModel(LightningModule):
    def __init__(self, torch_model: torch.nn.Module):
        super().__init__()
        self.model = torch_model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_losses(self, batch):
        global er_pos_weight, pr_pos_weight
        out = self.model(batch.x, batch.edge_index, batch.batch)
        er_labels = batch.y[:, 0]
        batch_er_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out[:, 0],
            er_labels,
            pos_weight=er_pos_weight,
        )
        pr_labels = batch.y[:, 1]
        batch_pr_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out[:, 1], pr_labels, pos_weight=pr_pos_weight
        )
        loss = batch_er_loss + batch_pr_loss
        return batch_er_loss, batch_pr_loss, loss

    def training_step(self, batch):
        batch_er_loss, batch_pr_loss, loss = self.compute_losses(batch)
        self.log(
            "train_er_loss",
            batch_er_loss,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train_pr_loss",
            batch_pr_loss,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch):
        batch_er_loss, batch_pr_loss, loss = self.compute_losses(batch)
        self.log(
            "val_er_loss",
            batch_er_loss,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_pr_loss",
            batch_pr_loss,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )
        self.log("val_loss", loss, batch_size=len(batch), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


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

        global er_pos_weight, pr_pos_weight
        er_pos_weight = torch.tensor(
            sum((labels[0] == 0 for labels in self.y))
            / sum((labels[0] == 1 for labels in self.y))
        )
        pr_pos_weight = torch.tensor(
            sum((labels[1] == 0 for labels in self.y))
            / sum((labels[1] == 1 for labels in self.y))
        )

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

            model = LightningModel(make_model())
            checkpoint = ModelCheckpoint(
                monitor="val_loss", mode="min", dirpath="checkpoints", filename="best"
            )
            logger = CSVLogger(save_dir="logs", name=model_name, version=fold_num)
            trainer = Trainer(
                accelerator="gpu",
                accumulate_grad_batches=1024,  # i.e. all the batches
                callbacks=[checkpoint],
                enable_progress_bar=False,
                logger=logger,
                max_epochs=epochs,
            )
            train_loader = torch_geometric.loader.DataLoader(
                torch.utils.data.Subset(self.dataset, train_idxs),
                batch_size=batch_size,
                shuffle=True,
            )
            validation_loader = torch_geometric.loader.DataLoader(
                torch.utils.data.Subset(self.dataset, validation_idxs),
                batch_size=batch_size,
            )

            t0 = time()
            trainer.fit(model, train_loader, validation_loader)
            t1 = time()

            train_times[fold_num] = t1 - t0
            with open(f"{model_name}_ER.metrics", "a") as f:
                f.write(f"Train time: {t1 - t0}\n")
            with open(f"{model_name}_PR.metrics", "a") as f:
                f.write(f"Train time: {t1 - t0}\n")

            ckpt = torch.load(checkpoint.best_model_path)
            logs = pd.read_csv(f"logs/{model_name}/version_{fold_num}/metrics.csv")
            epochs_list = range(1, epochs + 1)
            fig, ax = plt.subplots()
            ax.plot(epochs_list, logs["train_er_loss"].dropna(), label="train")
            ax.plot(epochs_list, logs["val_er_loss"].dropna(), label="validation")
            ax.plot(
                ckpt["epoch"] + 1,
                [
                    log["val_er_loss"]
                    for log in logs.to_dict("records")
                    if log["epoch"] == ckpt["epoch"] and not pd.isna(log["val_er_loss"])
                ],
                marker="o",
            )
            fig.legend()
            fig.tight_layout()
            fig.savefig(f"{model_name}_loss_ER_{fold_num}.png")
            ax.clear()
            ax.plot(epochs_list, logs["train_pr_loss"].dropna(), label="train")
            ax.plot(epochs_list, logs["val_pr_loss"].dropna(), label="validation")
            ax.plot(
                ckpt["epoch"] + 1,
                [
                    log["val_pr_loss"]
                    for log in logs.to_dict("records")
                    if log["epoch"] == ckpt["epoch"] and not pd.isna(log["val_pr_loss"])
                ],
                marker="o",
            )
            fig.legend()
            fig.tight_layout()
            fig.savefig(f"{model_name}_loss_PR_{fold_num}.png")

            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            y_pred: list[torch.Tensor] = []
            y_true: list[torch.Tensor] = []
            with torch.no_grad():
                for batch_data in validation_loader:
                    y_pred.extend(
                        torch.nn.functional.sigmoid(
                            model(batch_data.x, batch_data.edge_index, batch_data.batch)
                        )
                    )
                    y_true.extend(batch_data.y)
            for label_idx, label in enumerate(("ER", "PR")):
                metric_idx = 0
                y_pred_label = [round(t[label_idx].item()) for t in y_pred]
                y_true_label = [round(t[label_idx].item()) for t in y_true]
                for _, l_metric_func in LABEL_METRICS:
                    scores[label_idx, fold_num, metric_idx] = l_metric_func(
                        y_true_label, y_pred_label
                    )
                    metric_idx += 1
                y_pred_prob = [float(t[label_idx].item()) for t in y_pred]
                y_true_prob = [int(t[label_idx].item()) for t in y_true]
                for _, p_metric_func in PROBABILITY_METRICS:
                    scores[label_idx, fold_num, metric_idx] = p_metric_func(
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
