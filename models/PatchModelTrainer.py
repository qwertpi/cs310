from __future__ import annotations
from functools import partial
import pickle
from time import time
from typing import TYPE_CHECKING, Callable, ParamSpec

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat
import numpy as np
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.typing import Tensor  # type: ignore

if torch.cuda.is_available():
    import cupy as np  # type: ignore
else:
    import numpy as np

P = ParamSpec("P")


def float_wrapper(f: Callable[P, ConvertibleToFloat]):
    def ret_func(*args: P.args, **kwargs: P.kwargs):
        return float(f(*args, **kwargs))

    return ret_func


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
        make_model: Callable[[], Pipeline],
        aggregator: Callable[[np.ndarray], int],
        model_name: str,
        label_idx: int,
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

        scores = np.empty((NUM_FOLDS, len(ALL_METRICS)))
        train_times = np.empty((NUM_FOLDS))
        folds = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True).split(
            self.graphs, self.y_compact, self.groups
        )
        # Delete the file if it already exists
        with open(f"{model_name}.metrics", "w") as f:
            f.write("")
        for fold_num, (train_idxs, validation_idxs) in enumerate(folds):
            with open(f"{model_name}.metrics", "a") as f:
                f.write(f"{fold_num}\n")
            metric_idx: int = 0

            train_x: list[Tensor] = []
            train_y: list[int] = []
            for i in train_idxs:
                for node_features in self.graphs[i].x:
                    train_x.append(node_features)
                    train_y.append(self.graph_level_y[i][label_idx])

            t0 = time()
            model = make_model()
            model.fit(np.array(train_x), np.array(train_y))
            t1 = time()
            train_times[fold_num] = t1 - t0
            with open(f"{model_name}.metrics", "a") as f:
                f.write(f"Train time: {t1 - t0}\n")

            graph_level_y_pred = []
            graph_level_y_true: list[int] = []
            for i in validation_idxs:
                graph_level_y_pred.append(
                    aggregator(model.predict_proba(np.array(self.graphs[i].x)))
                )
                graph_level_y_true.append(self.graph_level_y[i][label_idx])
            for _, metric_func in LABEL_METRICS:
                scores[fold_num, metric_idx] = metric_func(
                    graph_level_y_true, graph_level_y_pred
                )
                metric_idx += 1

            node_level_x = []
            node_level_y_true: list[int] = []
            for i in validation_idxs:
                for node_features in self.graphs[i].x:
                    node_level_x.append(node_features)
                    node_level_y_true.append(self.graph_level_y[i][label_idx])
            node_level_y_pred = model.predict_proba(np.array(node_level_x))[:, 1]
            for _, metric_func in PROBABILITY_METRICS:  # type: ignore
                scores[fold_num, metric_idx] = metric_func(
                    node_level_y_true, node_level_y_pred
                )
                metric_idx += 1

            with open(f"{model_name}.metrics", "a") as f:
                for (metric_name, _), metric_val in zip(ALL_METRICS, scores[fold_num]):
                    f.write(f"{metric_name}: {metric_val}\n")

        means = np.mean(scores, axis=0)
        std_devs = np.std(scores, axis=0)
        with open(f"{model_name}.metrics", "a") as f:
            f.write("MEAN\n")
            for (metric_name, _), average in zip(ALL_METRICS, means):
                f.write(f"{metric_name}: {average}\n")
            f.write(f"Train time: {np.mean(train_times)}\n")

            f.write("STD_DEV\n")
            for (metric_name, _), var in zip(ALL_METRICS, std_devs):
                f.write(f"{metric_name}: {var}\n")
            f.write(f"Train time: {np.std(train_times)}\n")
