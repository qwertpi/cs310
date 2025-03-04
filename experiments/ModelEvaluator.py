from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Callable, ParamSpec, TextIO, TypeVar

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
import torch

# ABCTB stats
ER_POS_PREVALANCE = 2001 / 2538
ER_POS_GIVEN_PR_POS_PREVALANCE = 1743 / 1792
ER_POS_GIVEN_PR_NEG_PREVALANCE = 258 / 746
PR_POS_PREVALANCE = 1792 / 2538
PR_POS_GIVEN_ER_POS_PREVALANCE = 1743 / 2001
PR_POS_GIVEN_ER_NEG_PREVALANCE = 49 / 537

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


def point_usable_information(
    true_labels: torch.Tensor, predictions: torch.Tensor, null_distribution: float
):
    model_entropy = torch.log2(
        torch.where(true_labels == 1, predictions, 1 - predictions)
    )
    null_entropy = torch.log2(
        torch.where(true_labels == 1, null_distribution, 1 - null_distribution)
    )
    return model_entropy - null_entropy


def average_usable_information(
    true_labels: list[int],
    predictions: list[float],
    null_distribution: float,
):
    return point_usable_information(
        torch.tensor(true_labels), torch.tensor(predictions), null_distribution
    ).mean()


def scatter_plots(
    predictions_list: list[tuple[float, float]],
    true_labels_list: list[tuple[int, int]],
    prefix: str,
):
    def plot(
        label_idx: int,
        suffix: str,
        erpos_prneg_color: str,
        erpos_prneg_marker: str,
        erneg_prpos_color: str,
        erneg_prpos_marker: str,
    ):
        y = np.array(predictions_list)[:, label_idx]
        sort_idxs = np.argsort(y)
        y = y[sort_idxs]
        true_labels = np.array(true_labels_list)[sort_idxs]
        x = np.empty_like(y)
        GAP = 0.05
        last_y: dict[float, float] = {}
        for i in range(len(y)):
            candidate_x = 0.0
            while last_y.get(candidate_x) and abs(y[i] - last_y[candidate_x]) < GAP:
                candidate_x += GAP
            x[i] = candidate_x
            last_y[candidate_x] = y[i]

        fig, ax = plt.subplots()

        def scatter(
            er_target: bool, pr_target: bool, marker: str, color: str, label: str
        ):
            mask = (true_labels[:, 0] == er_target) & (true_labels[:, 1] == pr_target)  # noqa: E712
            ax.scatter(
                x[mask],
                y[mask],
                marker=marker,
                color=color,
                label=label,
            )

        scatter(True, True, "o", "green", "ER+PR+")
        scatter(True, False, erpos_prneg_marker, erpos_prneg_color, "ER+PR-")
        scatter(False, True, erneg_prpos_marker, erneg_prpos_color, "ER-PR+")
        scatter(False, False, "s", "red", "ER-PR-")

        ax.set_xticks([])
        fig.tight_layout()
        ax.legend()
        fig.savefig(f"{prefix}_scatter_{suffix}.png")
        plt.close(fig)

    plot(0, "ER", "green", "s", "red", "o")
    plot(1, "PR", "red", "o", "green", "s")


class ModelEvaluator:
    METRICS: list[
        tuple[str, Callable[[list[int], list[float], list[int], list[float]], float]]
    ] = [
        ("AUC_ROC_ER", fstify2a(typed_roc)),
        ("AUC_PR_ER", fstify2a(typed_pr)),
        ("AUC_ROC_PR", sndify2a(typed_roc)),
        ("AUC_PR_PR", sndify2a(typed_pr)),
        ("AUC_ROC_LABELSAGREE", roc_labels_agree),
        (
            "UI_ER",
            fstify2a(
                partial(average_usable_information, null_distribution=ER_POS_PREVALANCE)
            ),
        ),
        (
            "UI_PR",
            sndify2a(
                partial(average_usable_information, null_distribution=PR_POS_PREVALANCE)
            ),
        ),
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
            "AUC_ROC_ER|PR+",
            condition_metric_wrapper(fstify2a(typed_roc), is_pr_pos),
        ),
        (
            "AUC_ROC_ER|PR-",
            condition_metric_wrapper(fstify2a(typed_roc), is_pr_neg),
        ),
        (
            "AUC_ROC_PR|ER+",
            condition_metric_wrapper(sndify2a(typed_roc), is_er_pos),
        ),
        (
            "AUC_ROC_PR|ER-",
            condition_metric_wrapper(sndify2a(typed_roc), is_er_neg),
        ),
        (
            "UI_ER|PR+",
            condition_metric_wrapper(
                fstify2a(
                    partial(
                        average_usable_information,
                        null_distribution=ER_POS_GIVEN_PR_POS_PREVALANCE,
                    )
                ),
                is_pr_pos,
            ),
        ),
        (
            "UI_ER|PR-",
            condition_metric_wrapper(
                fstify2a(
                    partial(
                        average_usable_information,
                        null_distribution=ER_POS_GIVEN_PR_NEG_PREVALANCE,
                    )
                ),
                is_pr_neg,
            ),
        ),
        (
            "UI_PR|ER+",
            condition_metric_wrapper(
                sndify2a(
                    partial(
                        average_usable_information,
                        null_distribution=PR_POS_GIVEN_ER_POS_PREVALANCE,
                    )
                ),
                is_er_pos,
            ),
        ),
        (
            "UI_PR|ER-",
            condition_metric_wrapper(
                sndify2a(
                    partial(
                        average_usable_information,
                        null_distribution=PR_POS_GIVEN_ER_NEG_PREVALANCE,
                    )
                ),
                is_er_neg,
            ),
        ),
    ]
    ALL_METRICS = METRICS + CONDITIONED_METRICS

    DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC = ("Confusion_ERPR", erpr_confusion_matrix)

    def __init__(self, num_folds: int, file: TextIO):
        self.fold_num = 0
        self.scores = np.empty((num_folds, len(self.ALL_METRICS)))
        self.double_receptor_confusion_matrices = np.empty((num_folds, 4, 4))
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
        scatter_plots(
            list(zip(y_pred_er, y_pred_pr)),
            list(zip(y_true_er, y_true_pr)),
            f"{self.file.name.split('.')[0]}_{self.fold_num}",
        )
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

        self.double_receptor_confusion_matrices[self.fold_num] = (
            self.DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC[
                1
            ](y_true_er, y_pred_er, y_true_pr, y_pred_pr)
        )

        for (metric_name, _), metric_val in zip(
            self.ALL_METRICS, self.scores[self.fold_num]
        ):
            self.file.write(f"{metric_name}: {metric_val}\n")

        self.file.write(
            f"{self.DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC[0]}: {self.double_receptor_confusion_matrices[self.fold_num]}\n"
        )

        self.file.flush()
        self.fold_num += 1

    def close(self):
        metric_means = np.nanmean(self.scores, axis=0)
        double_receptor_matrix_mean = np.mean(
            self.double_receptor_confusion_matrices, axis=0
        )
        metric_std_devs = np.nanstd(self.scores, axis=0)
        double_receptor_matrix_std_dev = np.std(
            self.double_receptor_confusion_matrices, axis=0
        )
        self.file.write("MEAN\n")
        for (metric_name, _), mean in zip(self.ALL_METRICS, metric_means):
            self.file.write(f"{metric_name}: {mean}\n")
        self.file.write(
            f"{self.DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC[0]}: {double_receptor_matrix_mean}\n"
        )
        self.file.write(f"Train time: {np.mean(self.train_times)}\n")
        self.file.write(f"Epochs: {np.mean(self.epochs)}\n")

        self.file.write("STD_DEV\n")
        for (metric_name, _), stddev in zip(self.ALL_METRICS, metric_std_devs):
            self.file.write(f"{metric_name}: {stddev}\n")
        self.file.write(
            f"{self.DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC[0]}: {double_receptor_matrix_std_dev}\n"
        )
        self.file.write(f"Train time: {np.std(self.train_times)}\n")
        self.file.write(f"Epochs: {np.std(self.epochs)}\n")
        self.file.close()
