import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score  # type: ignore
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.typing import Tensor  # type: ignore

with open("../../data/train_data.pkl", "rb") as f:
    data: list[tuple[str, Data, tuple[bool, bool]]] = pickle.load(f)

groups: list[str] = []
x: list[Tensor] = []
graphs: list[Data] = []
graph_level_y: list[tuple[bool, bool]] = []
y_compact: list[str] = []
for group, graph, graph_label in data:
    y_compact.append(str(int(graph_label[0])) + str(int(graph_label[1])))
    graph_level_y.append(graph_label)
    graphs.append(graph)
    groups.append(group)

for model_name, label_pos in (("ER", 0), ("PR", 1)):
    print(model_name)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True)
    num = 0
    for train_idxs, validation_idxs in splitter.split(graphs, y_compact, groups):
        print(num)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                solver="newton-cholesky",
                max_iter=50000,
                n_jobs=-1,
            ),
        )
        train_x = []
        train_y = []
        for i in train_idxs:
            for node_features in graphs[i].x:
                train_x.append(node_features)
                train_y.append(graph_level_y[i][label_pos])
        model.fit(train_x, train_y)
        graph_level_y_pred_mean = []
        graph_level_y_pred_mode = []
        graph_level_y_true = []
        for i in validation_idxs:
            graph_level_y_pred_mean.append(
                np.argmax(
                    np.mean(
                        model.predict_proba(graphs[i].x), axis=0
                    )
                )
            )
            graph_level_y_pred_mode.append(
                round(np.mean(
                    np.argmax(
                        model.predict_proba(graphs[i].x), axis=1
                    )
                ))
            )
            graph_level_y_true.append(graph_level_y[i][label_pos])
        print("Acc: ", balanced_accuracy_score(graph_level_y_true, graph_level_y_pred_mean))
        print("Acc: ", balanced_accuracy_score(graph_level_y_true, graph_level_y_pred_mode))

        validation_x = []
        validation_y_true = []
        for i in validation_idxs:
            for node_features in graphs[i].x:
                validation_x.append(node_features)
                validation_y_true.append(graph_level_y[i][label_pos])

        y_pred = model.predict_proba(validation_x)[:, 1]
        print("AUC_ROC: ", roc_auc_score(validation_y_true, y_pred))
        print("AUC_PR: ", average_precision_score(validation_y_true, y_pred), flush=True)
        with open(f"{model_name}_{num}.pkl", "wb") as f:
            pickle.dump(model, f)
        num += 1
