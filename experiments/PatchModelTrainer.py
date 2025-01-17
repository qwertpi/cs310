from __future__ import annotations
import pickle
from time import time
from typing import Callable

from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
import torch

if torch.cuda.is_available():
    import cupy as np  # type: ignore
else:
    import numpy as np
from torch_geometric.data import Data  # type: ignore
from torch_geometric.typing import Tensor  # type: ignore
from tqdm import tqdm

from ModelEvaluator import ModelEvaluator


class PatchModelTrainer:
    def __init__(self):
        with open("../../../data/train_data.pkl", "rb") as f:
            data: list[tuple[str, Data, tuple[bool, bool]]] = pickle.load(f)
        self.groups: list[str] = []
        self.x: list[Tensor] = []
        self.graphs: list[Data] = []
        self.graph_level_y: list[tuple[bool, bool]] = []
        self.y_compact: list[str] = []
        for group, graph, graph_label in data:
            self.y_compact.append(str(int(graph_label[0])) + str(int(graph_label[1])))
            self.graph_level_y.append(graph_label)
            self.graphs.append(graph)
            self.groups.append(group)

    def train_and_validate(
        self,
        make_er_model: Callable[[], Pipeline],
        make_pr_model: Callable[[], Pipeline],
        model_name: str,
    ):
        NUM_FOLDS = 5
        # Delete the file if it already exists
        with open(f"{model_name}.metrics", "w") as f:
            f.write("")
        folds = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True).split(
            self.graphs, self.y_compact, self.groups
        )
        evaluator = ModelEvaluator(NUM_FOLDS, open(f"{model_name}.metrics", "a"))
        for train_idxs, validation_idxs in tqdm(folds, total=NUM_FOLDS):
            train_x = [
                node_features for i in train_idxs for node_features in self.graphs[i].x
            ]
            train_y_er = [
                self.graph_level_y[i][0]
                for i in train_idxs
                for feat in self.graphs[i].x
            ]
            train_y_pr = [
                self.graph_level_y[i][1]
                for i in train_idxs
                for feat in self.graphs[i].x
            ]
            t0 = time()
            er_model = make_er_model()
            er_model.fit(np.array(train_x), np.array(train_y_er))
            pr_model = make_pr_model()
            pr_model.fit(np.array(train_x), np.array(train_y_pr))
            t1 = time()

            # The model predicts the negative class and the positive class (sums to 1), we are interested in the positive class
            y_pred_er: list[float] = [
                np.mean(
                    er_model.predict_proba(np.array(self.graphs[i].x))[:, 1], axis=0
                )
                for i in validation_idxs
            ]
            y_pred_pr: list[float] = [
                np.mean(
                    pr_model.predict_proba(np.array(self.graphs[i].x))[:, 1], axis=0
                )
                for i in validation_idxs
            ]
            y_true: list[list[int]] = [self.graph_level_y[i] for i in validation_idxs]
            evaluator.evalaute_fold(t1 - t0, -1, y_true, y_pred_er, y_pred_pr)
        evaluator.close()
