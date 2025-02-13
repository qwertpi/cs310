from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ParamSpec, TextIO, TypeVar

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat

import numpy as np
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

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


def is_er_pos(t: list[int]) -> bool:
    return t[0]  # type: ignore


def is_er_neg(t: list[int]) -> bool:
    return not t[0]  # type: ignore


def is_pr_pos(t: list[int]) -> bool:
    return t[1]  # type: ignore


def is_pr_neg(t: list[int]) -> bool:
    return not t[1]  # type: ignore


def typed_roc(x1: list[int], x2: list[float]) -> float:
    if len(set(x1)) < 2:
        return float("nan")
    return float_wrapper(roc_auc_score)(x1, x2)  # type: ignore


def typed_pr(x1: list[int], x2: list[float]) -> float:
    if len(set(x1)) < 2:
        return float("nan")
    return float_wrapper(average_precision_score)(x1, x2)  # type: ignore


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


def erpr_confusion_matrix(
    er_true: list[int], er_pred: list[float], pr_true: list[int], pr_pred: list[float]
):
    # The labels are stored as booleans but we want 0, 1 for this
    y_true_concat = [f"{int(er)}{int(pr)}" for er, pr in zip(er_true, pr_true)]
    y_pred_concat = [f"{round(er)}{round(pr)}" for er, pr in zip(er_pred, pr_pred)]
    return confusion_matrix(
        y_true_concat, y_pred_concat, labels=["00", "01", "10", "11"], normalize="true"
    )


def single_receptor_confusion_matrix(true: list[int], pred: list[float]):
    return confusion_matrix(
        [int(t) for t in true],
        [round(p) for p in pred],
        labels=[0, 1],
        normalize="true",
    )


class ModelEvaluator:
    METRICS: list[
        tuple[str, Callable[[list[int], list[float], list[int], list[float]], float]]
    ] = [
        ("AUC_ROC_ER", fstify2a(typed_roc)),
        ("AUC_PR_ER", fstify2a(typed_pr)),
        ("AUC_ROC_PR", sndify2a(typed_roc)),
        ("AUC_PR_PR", sndify2a(typed_pr)),
        ("AUC_ROC_LABELSAGREE", roc_labels_agree),
    ]
    CONDITIONED_METRICS: list[
        tuple[
            str,
            Callable[
                [
                    list[list[int]],
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
    ALL_METRICS = METRICS + CONDITIONED_METRICS

    DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC = ("Confusion_ERPR", erpr_confusion_matrix)
    SINGLE_RECEPETOR_CONFUSION_MATRICES_FUNCS = [
        ("Confusion_ER", fstify2a(single_receptor_confusion_matrix)),
        ("Confusion_PR", sndify2a(single_receptor_confusion_matrix)),
    ]

    def __init__(self, num_folds: int, file: TextIO):
        self.fold_num = 0
        self.scores = np.empty((num_folds, len(self.ALL_METRICS)))
        self.single_receptor_confusion_matrices = np.empty(
            (num_folds, len(self.SINGLE_RECEPETOR_CONFUSION_MATRICES_FUNCS), 2, 2)
        )
        self.double_receptor_confusion_matrices = np.empty((num_folds, 1, 4, 4))
        self.train_times = np.empty((num_folds))
        self.epochs = np.empty((num_folds))
        self.file = file

    def evalaute_fold(
        self,
        time: float,
        epochs: int,
        y_true: list[list[int]],
        y_pred_er: list[float],
        y_pred_pr: list[float],
    ):
        self.file.write(f"{self.fold_num}\n")
        self.train_times[self.fold_num] = time / 60
        self.file.write(f"Train time: {self.train_times[self.fold_num]}\n")
        self.epochs[self.fold_num] = epochs
        self.file.write(f"Epochs: {epochs}\n")

        y_true_er = [t[0] for t in y_true]
        y_true_pr = [t[1] for t in y_true]
        metric_idx = 0
        for _, p_metric_func in self.METRICS:
            self.scores[self.fold_num, metric_idx] = p_metric_func(
                y_true_er, y_pred_er, y_true_pr, y_pred_pr
            )
            metric_idx += 1
        for _, cp_metric_func in self.CONDITIONED_METRICS:
            self.scores[self.fold_num, metric_idx] = cp_metric_func(
                y_true, y_true_er, y_pred_er, y_true_pr, y_pred_pr
            )
            metric_idx += 1

        self.double_receptor_confusion_matrices[self.fold_num, 0] = (
            self.DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC[
                1
            ](y_true_er, y_pred_er, y_true_pr, y_pred_pr)
        )
        for i, (_, matrix_func) in enumerate(
            self.SINGLE_RECEPETOR_CONFUSION_MATRICES_FUNCS
        ):
            self.single_receptor_confusion_matrices[self.fold_num, i] = matrix_func(
                y_true_er, y_pred_er, y_true_pr, y_pred_pr
            )

        for (metric_name, _), metric_val in zip(
            self.ALL_METRICS, self.scores[self.fold_num]
        ):
            self.file.write(f"{metric_name}: {metric_val}\n")

        for (matrix_name, _), matrix in zip(
            self.SINGLE_RECEPETOR_CONFUSION_MATRICES_FUNCS,
            self.single_receptor_confusion_matrices[self.fold_num],
        ):
            self.file.write(f"{matrix_name}: {matrix}\n")
        for matrix in self.double_receptor_confusion_matrices[self.fold_num]:
            self.file.write(
                f"{self.DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC[0]}: {matrix}\n"
            )

        self.file.flush()
        self.fold_num += 1

    def close(self):
        metric_means = np.nanmean(self.scores, axis=0)
        single_receptor_matrix_means = np.mean(
            self.single_receptor_confusion_matrices, axis=0
        )
        double_receptor_matrix_means = np.mean(
            self.double_receptor_confusion_matrices, axis=0
        )
        metric_std_devs = np.nanstd(self.scores, axis=0)
        single_receptor_matrix_std_devs = np.std(
            self.single_receptor_confusion_matrices, axis=0
        )
        double_receptor_matrix_std_devs = np.std(
            self.double_receptor_confusion_matrices, axis=0
        )
        self.file.write("MEAN\n")
        for (metric_name, _), mean in zip(self.ALL_METRICS, metric_means):
            self.file.write(f"{metric_name}: {mean}\n")
        for (matrix_name, _), mean in zip(
            self.SINGLE_RECEPETOR_CONFUSION_MATRICES_FUNCS, single_receptor_matrix_means
        ):
            self.file.write(f"{matrix_name}: {mean}\n")
        for mean in double_receptor_matrix_means:
            self.file.write(
                f"{self.DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC[0]}: {mean}\n"
            )
        self.file.write(f"Train time: {np.mean(self.train_times)}\n")
        self.file.write(f"Epochs: {np.mean(self.epochs)}\n")

        self.file.write("STD_DEV\n")
        for (metric_name, _), stddev in zip(self.ALL_METRICS, metric_std_devs):
            self.file.write(f"{metric_name}: {stddev}\n")
        for (matrix_name, _), stddev in zip(
            self.SINGLE_RECEPETOR_CONFUSION_MATRICES_FUNCS,
            single_receptor_matrix_std_devs,
        ):
            self.file.write(f"{matrix_name}: {stddev}\n")
        for stddev in double_receptor_matrix_std_devs:
            self.file.write(
                f"{self.DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC[0]}: {stddev}\n"
            )
        self.file.write(f"Train time: {np.std(self.train_times)}\n")
        self.file.write(f"Epochs: {np.std(self.epochs)}\n")
        self.file.close()
