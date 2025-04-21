from __future__ import annotations
from enum import Enum, auto
from time import time
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
import torch

import torch_geometric.loader  # type: ignore

from ModelEvaluator import ModelEvaluator
from LightningModel import LightningModel
from TorchDatasets import ABCTBDataset, Dataset, TCGADataset

torch.set_float32_matmul_precision("medium")


class DataSource(Enum):
    ABCTB = auto()
    TCGA = auto()


class GNNModelTrainer:
    def __init__(self, datasource: DataSource):
        self.dataset: Dataset
        if datasource is DataSource.TCGA:
            self.dataset = TCGADataset()
            self.batch_size = 32
        elif datasource is DataSource.ABCTB:
            self.dataset = ABCTBDataset()
            self.batch_size = 2
        else:
            raise RuntimeError()

    def train_and_validate(
        self,
        make_model: Callable[[int], torch.nn.Module],
        model_name: str,
        remove_label_correlations: bool = False,
        discard_conflicting_labels: bool = False,
        hinge_loss: bool = False,
        spectral_decoupling: bool = False,
        weight_decay: float = 1e-2,  # AdamW's default value
        penalty_weight_er: Optional[float] = None,
        penalty_weight_pr: Optional[float] = None,
    ):
        # Delete the file if it already exists
        with open(f"{model_name}.metrics", "w") as f:
            f.write("")
        evaluator = ModelEvaluator(5, open(f"{model_name}.metrics", "a"))
        model = None
        for fold_num, (train_idxs, validation_idxs) in enumerate(
            self.dataset.get_folds()
        ):
            model = LightningModel(
                make_model(self.dataset.get_feat_dim()),
                weight_decay,
                remove_label_correlations,
                discard_conflicting_labels,
                hinge_loss,
                spectral_decoupling,
                penalty_weight_er,
                penalty_weight_pr,
            )
            if isinstance(self.dataset, ABCTBDataset):
                early_stopping = EarlyStopping(monitor="val_loss", patience=10)
            elif isinstance(self.dataset, TCGADataset):
                early_stopping = EarlyStopping(monitor="val_loss", patience=30)
            else:
                raise RuntimeError()
            logger = CSVLogger(save_dir="logs", name=model_name, version=fold_num)
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
                batch_size=self.batch_size,
            )
            validation_loader = torch_geometric.loader.DataLoader(
                torch.utils.data.Subset(self.dataset, validation_idxs),
                batch_size=self.batch_size,
            )
            t0 = time()
            trainer.fit(model, train_loader, validation_loader)
            t1 = time()

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

        if model is None:
            raise RuntimeError
        folds_weights = [
            torch.load(f"logs/{model_name}/version_{i}/checkpoints/best.ckpt")["state_dict"]
            for i in range(5)
        ]  # fmt: skip
        weights = {
            k: sum([d[k] for d in folds_weights]) / len(folds_weights)
            for k in folds_weights[0].keys()
        }
        model.load_state_dict(weights)
        model.eval()
        y_pred_er = []
        y_pred_pr = []
        y_true = []
        evaluator = ModelEvaluator(1, open(f"{model_name}_test.metrics", "a"))
        with torch.no_grad():
            for graph_data in torch_geometric.loader.DataLoader(
                self.dataset.get_test_data(),
                batch_size=self.batch_size,
            ):
                prediction = torch.nn.functional.sigmoid(
                    model(graph_data.x, graph_data.edge_index, graph_data.batch)
                )
                y_pred_er.extend([float(t[0]) for t in prediction])
                y_pred_pr.extend([float(t[1]) for t in prediction])
                y_true.extend(
                    [[round(t[0].item()), round(t[1].item())] for t in graph_data.y]
                )
        evaluator.evalaute_fold(-1, -1, y_true, y_pred_er, y_pred_pr)
        evaluator.close()
