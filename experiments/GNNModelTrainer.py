from __future__ import annotations
import pickle
from shutil import rmtree
from time import time
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
import torch

from torch_geometric.data import Data  # type: ignore
import torch_geometric.loader  # type: ignore

from ModelEvaluator import (
    ModelEvaluator,
    ER_POS_PREVALANCE,
    PR_POS_PREVALANCE,
)

torch.set_float32_matmul_precision("medium")


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
        weight_decay: float,
    ):
        super().__init__()
        self.model = torch_model
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_losses(self, batch):
        er_labels = batch.y[:, 0]
        pr_labels = batch.y[:, 1]
        out = self.model(batch.x, batch.edge_index, batch.batch)
        er_pred = out[:, 0]
        pr_pred = out[:, 1]

        # We drop ER-PR+ cases to avoid confusing the model
        er_labels_given_pr_pos = er_labels[(pr_labels == 1) & (er_labels == 1)]
        er_labels_given_pr_neg = er_labels[pr_labels == 0]
        pr_labels_given_er_pos = pr_labels[er_labels == 1]
        pr_labels_given_er_neg = pr_labels[(er_labels == 0) & (pr_labels == 0)]

        er_pred_given_pr_pos = er_pred[(pr_labels == 1) & (er_labels == 1)]
        er_pred_given_pr_neg = er_pred[pr_labels == 0]
        pr_pred_given_er_pos = pr_pred[er_labels == 1]
        pr_pred_given_er_neg = pr_pred[(er_labels == 0) & (pr_labels == 0)]

        # Oversampling in the BCE losses so positive and negative examples count equally
        er_pos_weight = torch.tensor(ER_POS_PREVALANCE)
        pr_pos_weight = torch.tensor(PR_POS_PREVALANCE)

        batch_er_loss_given_pr_pos = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                er_pred_given_pr_pos,
                er_labels_given_pr_pos,
                pos_weight=er_pos_weight,
            )
        ).nan_to_num(0)
        batch_er_loss_given_pr_neg = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                er_pred_given_pr_neg,
                er_labels_given_pr_neg,
                pos_weight=er_pos_weight,
            )
        ).nan_to_num(0)

        batch_pr_loss_given_er_pos = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                pr_pred_given_er_pos,
                pr_labels_given_er_pos,
                pos_weight=pr_pos_weight,
            )
        ).nan_to_num(0)
        batch_pr_loss_given_er_neg = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                pr_pred_given_er_neg,
                pr_labels_given_er_neg,
                pos_weight=pr_pos_weight,
            )
        ).nan_to_num(0)

        # Weight PR+, and PR- cases equally whereas if we had done
        # one big call to BCE they would have been skewed by the uneven
        # distribution of the PR label
        batch_er_loss = (
            0.5 * batch_er_loss_given_pr_pos + 0.5 * batch_er_loss_given_pr_neg
        )
        # Weight ER+, and ER- cases equally whereas if we had done
        # one big call to BCE they would have been skewed by the uneven
        # distribution of the ER label
        batch_pr_loss = (
            0.5 * batch_pr_loss_given_er_pos + 0.5 * batch_pr_loss_given_er_neg
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
        # PyTorch's default weight decay decays parameters that should not be regularised
        # This code based on: https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/14
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for module in self.modules():
            if isinstance(
                module,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                    torch.nn.PReLU,
                ),
            ):
                no_decay_params.extend(module.parameters(recurse=False))
            else:
                for name, param in module.named_parameters(recurse=False):
                    if "bias" in name:
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

        if sum((p.numel() for p in no_decay_params)) + sum(
            (p.numel() for p in decay_params)
        ) != sum((p.numel() for p in self.parameters(recurse=True))):
            raise RuntimeError("Lost or gained paramaters")

        return torch.optim.AdamW(
            (
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0},
            )
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

    def train_and_validate(
        self,
        make_model: Callable[[], torch.nn.Module],
        model_name: str,
        weight_decay: float = 1e-2,  # AdamW's default value
    ):
        NUM_FOLDS = 5
        # Delete the file if it already exists
        with open(f"{model_name}.metrics", "w") as f:
            f.write("")
        evaluator = ModelEvaluator(NUM_FOLDS, open(f"{model_name}.metrics", "a"))
        folds = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True).split(
            self.y_compact, self.y_compact, self.groups
        )
        batch_size = 1024
        for fold_num, (train_idxs, validation_idxs) in enumerate(folds):
            model = LightningModel(
                make_model(),
                weight_decay,
            )
            early_stopping = EarlyStopping(monitor="val_loss", patience=20)

            t0 = t1 = checkpoint = validation_loader = None
            while batch_size > 1:
                try:
                    logger = CSVLogger(
                        save_dir="logs", name=model_name, version=fold_num
                    )
                    checkpoint = ModelCheckpoint(monitor="val_loss", filename="best")
                    trainer = Trainer(
                        accelerator="gpu",
                        accumulate_grad_batches=4096,  # i.e. all the batches
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
            if (
                t0 is None
                or t1 is None
                or checkpoint is None
                or validation_loader is None
            ):
                return

            ckpt = torch.load(checkpoint.best_model_path)
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
            y_pred_er: list[float] = []
            y_pred_pr: list[float] = []
            y_true: list[list[int]] = []
            with torch.no_grad():
                for batch_data in validation_loader:
                    prediction = torch.nn.functional.sigmoid(
                        model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    )
                    y_pred_er.extend([float(t[0]) for t in prediction])
                    y_pred_pr.extend([float(t[1]) for t in prediction])
                    y_true.extend(
                        [[round(t[0].item()), round(t[1].item())] for t in batch_data.y]
                    )
            evaluator.evalaute_fold(
                t1 - t0, ckpt["epoch"], y_true, y_pred_er, y_pred_pr
            )
        evaluator.close()
