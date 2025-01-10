from __future__ import annotations
import pickle
from shutil import rmtree
from time import time
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
from sklearn.metrics import (  # type: ignore
    average_precision_score,
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


T = TypeVar("T")


def condition_metric_wrapper(
    f: Callable[[list[int], list[float], list[int], list[float]], float],
    filter: Callable[[T], bool],
):
    def ret_func(
        labels: list[T], x1: list[int], y1: list[float], x2: list[int], y2: list[float]
    ):
        idxs = [i for i, l in enumerate(labels) if filter(l)]
        return f(
            [x1[i] for i in idxs],
            [y1[i] for i in idxs],
            [x2[i] for i in idxs],
            [y2[i] for i in idxs],
        )

    return ret_func


def is_er_pos(t: torch.Tensor) -> bool:
    return t[0]  # type: ignore


def is_er_neg(t: torch.Tensor) -> bool:
    return not t[0]  # type: ignore


def is_pr_pos(t: torch.Tensor) -> bool:
    return t[1]  # type: ignore


def is_pr_neg(t: torch.Tensor) -> bool:
    return not t[1]  # type: ignore


def typed_roc(x1: list[int], x2: list[float]) -> float:
    if len(set(x1)) < 2:
        return float("nan")
    return float_wrapper(roc_auc_score)(x1, x2)  # type: ignore


def typed_pr(x1: list[int], x2: list[float]) -> float:
    if len(set(x1)) < 2:
        return float("nan")
    return float_wrapper(average_precision_score)(x1, x2)  # type: ignore


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
    def __init__(
        self,
        torch_model: torch.nn.Module,
        er_pos_weight: torch.Tensor,
        pr_pos_weight: torch.Tensor,
        weight_decay: float,
    ):
        super().__init__()
        self.model = torch_model
        self.er_pos_weight = er_pos_weight
        self.pr_pos_weight = pr_pos_weight
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_losses(self, batch):
        global er_pos_weight, pr_pos_weight
        out = self.model(batch.x, batch.edge_index, batch.batch)
        er_labels = batch.y[:, 0]
        batch_er_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out[:, 0],
            er_labels,
            pos_weight=self.er_pos_weight,
        )
        pr_labels = batch.y[:, 1]
        batch_pr_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out[:, 1], pr_labels, pos_weight=self.pr_pos_weight
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
        return torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay)


def fstify2a(f):
    def inner(x1, y1, x2, y2, **kwargs):
        return f(x1, y1, **kwargs)

    return inner


def sndify2a(f):
    def inner(x1, y1, x2, y2, **kwargs):
        return f(x2, y2, **kwargs)

    return inner


def roc_labels_agree(
    er_true: list[int], er_pred: list[float], pr_true: list[int], pr_pred: list[float]
):
    return typed_roc(
        [e == p for e, p in zip(er_true, pr_true)],
        [e * p + (1 - e) * (1 - p) for e, p in zip(er_pred, pr_pred)],
    )


class GNNModelTrainer:
    def __init__(self):
        with open("../../../data/train_data.pkl", "rb") as f:
            data: list[tuple[str, Data, tuple[bool, bool]]] = pickle.load(f)
        self.dataset = Dataset(data)
        self.groups: list[str] = []
        self.y: list[tuple[bool, bool]] = []
        self.y_compact: list[str] = []
        for group, _, graph_label in data:
            self.y_compact.append(str(int(graph_label[0])) + str(int(graph_label[1])))
            self.y.append(graph_label)
            self.groups.append(group)

        self.er_pos_weight = torch.tensor(
            sum((labels[0] == 0 for labels in self.y))
            / sum((labels[0] == 1 for labels in self.y))
        )
        self.pr_pos_weight = torch.tensor(
            sum((labels[1] == 0 for labels in self.y))
            / sum((labels[1] == 1 for labels in self.y))
        )

    def train_and_validate(
        self,
        make_model: Callable[[], torch.nn.Module],
        model_name: str,
        weight_decay: float,
    ):
        PROBABILITY_METRICS: list[
            tuple[
                str, Callable[[list[int], list[float], list[int], list[float]], float]
            ]
        ] = [
            ("AUC_ROC_ER", fstify2a(typed_roc)),
            ("AUC_PR_ER", fstify2a(typed_pr)),
            ("AUC_ROC_PR", sndify2a(typed_roc)),
            ("AUC_PR_PR", sndify2a(typed_pr)),
            ("AUC_ROC_LABELSAGREE", roc_labels_agree),
        ]
        CONDITIONED_PROBABILITY_METRICS: list[
            tuple[
                str,
                Callable[
                    [
                        list[torch.Tensor],
                        list[int],
                        list[float],
                        list[int],
                        list[float],
                    ],
                    float,
                ],
            ]
        ] = [
            (
                "AUC_ROC_ER_PR+",
                condition_metric_wrapper(fstify2a(typed_roc), is_pr_pos),
            ),
            (
                "AUC_ROC_ER_PR-",
                condition_metric_wrapper(fstify2a(typed_roc), is_pr_neg),
            ),
            (
                "AUC_ROC_PR_ER+",
                condition_metric_wrapper(sndify2a(typed_roc), is_er_pos),
            ),
            (
                "AUC_ROC_PR_ER-",
                condition_metric_wrapper(sndify2a(typed_roc), is_er_neg),
            ),
        ]
        ALL_METRICS = PROBABILITY_METRICS + CONDITIONED_PROBABILITY_METRICS

        NUM_FOLDS = 5
        scores = np.empty((NUM_FOLDS, len(ALL_METRICS)))
        num_epochs_used = np.empty((NUM_FOLDS))
        train_times = np.empty((NUM_FOLDS))
        folds = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True).split(
            self.y_compact, self.y_compact, self.groups
        )
        # Delete the file if it already exists
        with open(f"{model_name}.metrics", "w") as f:
            f.write("")
        batch_size = 1024
        for fold_num, (train_idxs, validation_idxs) in enumerate(folds):
            with open(f"{model_name}.metrics", "a") as f:
                f.write(f"{fold_num}\n")

            model = LightningModel(
                make_model(), self.er_pos_weight, self.pr_pos_weight, weight_decay
            )
            early_stopping = EarlyStopping(monitor="val_loss", patience=50)

            t0 = t1 = checkpoint = None
            while batch_size > 1:
                try:
                    logger = CSVLogger(
                        save_dir="logs", name=model_name, version=fold_num
                    )
                    checkpoint = ModelCheckpoint(monitor="val_loss", filename="best")
                    trainer = Trainer(
                        accelerator="gpu",
                        accumulate_grad_batches=1024,  # i.e. all the batches
                        callbacks=[checkpoint, early_stopping],
                        enable_progress_bar=False,
                        logger=logger,
                        max_epochs=-1,
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
                    break
                except torch.cuda.OutOfMemoryError:
                    batch_size //= 2
                    rmtree(f"logs/{model_name}/version_{fold_num}")
            if t0 is None or t1 is None or checkpoint is None:
                return

            train_times[fold_num] = t1 - t0
            with open(f"{model_name}.metrics", "a") as f:
                f.write(f"Train time: {t1 - t0}\n")

            ckpt = torch.load(checkpoint.best_model_path)
            num_epochs_used[fold_num] = ckpt["epoch"]
            with open(f"{model_name}.metrics", "a") as f:
                f.write(f"Epochs: {ckpt['epoch']}\n")
            logs = pd.read_csv(f"logs/{model_name}/version_{fold_num}/metrics.csv")
            epochs_list = range(logs["epoch"].min(), logs["epoch"].max() + 1)
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
            plt.close(fig)

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
            metric_idx = 0
            y_pred_er = [float(t[0].item()) for t in y_pred]
            y_pred_pr = [float(t[1].item()) for t in y_pred]
            y_true_er = [round(t[0].item()) for t in y_true]
            y_true_pr = [round(t[1].item()) for t in y_true]
            for _, p_metric_func in PROBABILITY_METRICS:
                scores[fold_num, metric_idx] = p_metric_func(
                    y_true_er, y_pred_er, y_true_pr, y_pred_pr
                )
                metric_idx += 1
            for n, (_, cp_metric_func) in enumerate(CONDITIONED_PROBABILITY_METRICS):
                scores[fold_num, metric_idx] = cp_metric_func(
                    y_true, y_true_er, y_pred_er, y_true_pr, y_pred_pr
                )
                metric_idx += 1

            with open(f"{model_name}.metrics", "a") as f:
                for (metric_name, _), metric_val in zip(ALL_METRICS, scores[fold_num]):
                    f.write(f"{metric_name}: {metric_val}\n")
        means = np.nanmean(scores, axis=0)
        std_devs = np.nanstd(scores, axis=0)
        with open(f"{model_name}.metrics", "a") as f:
            f.write("MEAN\n")
            for (metric_name, _), average in zip(ALL_METRICS, means):
                f.write(f"{metric_name}: {average}\n")
            f.write(f"Train time: {np.mean(train_times)}\n")
            f.write(f"Epochs: {np.mean(num_epochs_used)}\n")

            f.write("STD_DEV\n")
            for (metric_name, _), var in zip(ALL_METRICS, std_devs):
                f.write(f"{metric_name}: {var}\n")
            f.write(f"Train time: {np.std(train_times)}\n")
            f.write(f"Epochs: {np.std(num_epochs_used)}\n")
