from __future__ import annotations
from typing import Callable, TextIO, cast

from numpy.typing import NDArray

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (  # type: ignore
    auc,
)

from ModelEvaluationUtils import (
    condition_metric_wrapper,
    condition_wrapper,
    erpr_confusion_matrix,
    fstify2a,
    is_er_pos,
    is_pr_neg,
    sndify2a,
    typed_auc_roc,
    typed_roc_curve,
)


class ModelEvaluator:
    METRICS: list[
        tuple[str, Callable[[list[int], list[float], list[int], list[float]], float]]
    ] = [
        ("AUC_ROC_ER", fstify2a(typed_auc_roc)),
        ("AUC_ROC_PR", sndify2a(typed_auc_roc)),
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
            "AUC_ROC_ER|PR-",
            condition_metric_wrapper(fstify2a(typed_auc_roc), is_pr_neg),
        ),
        (
            "AUC_ROC_PR|ER+",
            condition_metric_wrapper(sndify2a(typed_auc_roc), is_er_pos),
        ),
    ]
    ALL_METRICS = METRICS + CONDITIONED_METRICS

    DOUBLE_RECEPTOR_CONFUSION_MATRIX_FUNC = ("Confusion_ERPR", erpr_confusion_matrix)

    def __init__(self, num_folds: int, file: TextIO):
        self.fold_num = 0

        self.er_roc_curve_fig, self.er_roc_curve_ax = plt.subplots()
        self.er_prneg_roc_curve_fig, self.er_prneg_roc_curve_ax = plt.subplots()
        self.pr_roc_curve_fig, self.pr_roc_curve_ax = plt.subplots()
        self.pr_erpos_roc_curve_fig, self.pr_erpos_roc_curve_ax = plt.subplots()
        self.er_fprs: list[NDArray] = []
        self.er_tprs: list[NDArray] = []
        self.er_prneg_fprs: list[NDArray] = []
        self.er_prneg_tprs: list[NDArray] = []
        self.pr_fprs: list[NDArray] = []
        self.pr_tprs: list[NDArray] = []
        self.pr_erpos_fprs: list[NDArray] = []
        self.pr_erpos_tprs: list[NDArray] = []

        self.er_auc_rocs = np.empty(num_folds)
        self.er_prneg_auc_rocs = np.empty(num_folds)
        self.pr_auc_rocs = np.empty(num_folds)
        self.pr_erpos_auc_rocs = np.empty(num_folds)

        self.scores = np.empty((num_folds, len(self.ALL_METRICS)))
        self.double_receptor_confusion_matrices = np.empty((num_folds, 4, 4))
        self.train_times = np.empty((num_folds))
        self.epochs = np.empty((num_folds))
        self.file = file

    def _scatter_plots(
        self,
        predictions_list: list[tuple[float, float]],
        true_labels_list: list[tuple[int, int]],
    ):
        prefix = f"{self.file.name.split('.')[0]}_{self.fold_num}"

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
                mask = (true_labels[:, 0] == er_target) & (
                    true_labels[:, 1] == pr_target
                )  # noqa: E712
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

    def _plot_rocs(
        self,
        y_true: list[list[int]],
        y_true_er: list[int],
        y_pred_er: list[float],
        y_true_pr: list[int],
        y_pred_pr: list[float],
    ):
        # Based on: https://github.com/foxtrotmike/CS909/blob/master/evaluation_example.ipynb
        # Date acessed: 2025-03-04
        er_fpr, er_tpr, _ = typed_roc_curve(y_true_er, y_pred_er)
        er_prneg_fpr, er_prneg_tpr, _ = cast(
            tuple[NDArray, NDArray, NDArray],
            condition_wrapper(typed_roc_curve, is_pr_neg)(y_true, y_true_er, y_pred_er),
        )
        pr_fpr, pr_tpr, _ = typed_roc_curve(y_true_pr, y_pred_pr)
        pr_erpos_fpr, pr_erpos_tpr, _ = cast(
            tuple[NDArray, NDArray, NDArray],
            condition_wrapper(typed_roc_curve, is_er_pos)(y_true, y_true_pr, y_pred_pr),
        )

        self.er_auc_rocs[self.fold_num] = auc(er_fpr, er_tpr)
        self.er_prneg_auc_rocs[self.fold_num] = auc(er_prneg_fpr, er_prneg_tpr)
        self.pr_auc_rocs[self.fold_num] = auc(pr_fpr, pr_tpr)
        self.pr_erpos_auc_rocs[self.fold_num] = auc(pr_erpos_fpr, pr_erpos_tpr)

        self.er_roc_curve_ax.plot(
            er_fpr,
            er_tpr,
            alpha=0.5,
            label=f"Fold {self.fold_num} (AUC = {self.er_auc_rocs[self.fold_num]:.2f})",
        )
        self.er_prneg_roc_curve_ax.plot(
            er_prneg_fpr,
            er_prneg_tpr,
            alpha=0.5,
            label=f"Fold {self.fold_num} (AUC = {self.er_prneg_auc_rocs[self.fold_num]:.2f})",
        )
        self.pr_roc_curve_ax.plot(
            pr_fpr,
            pr_tpr,
            alpha=0.5,
            label=f"Fold {self.fold_num} (AUC = {self.pr_auc_rocs[self.fold_num]:.2f})",
        )
        self.pr_erpos_roc_curve_ax.plot(
            pr_erpos_fpr,
            pr_erpos_tpr,
            alpha=0.5,
            label=f"Fold {self.fold_num} (AUC = {self.pr_erpos_auc_rocs[self.fold_num]:.2f})",
        )

        self.er_fprs.append(er_fpr)
        self.er_tprs.append(er_tpr)
        self.er_prneg_fprs.append(er_prneg_fpr)
        self.er_prneg_tprs.append(er_prneg_tpr)
        self.pr_fprs.append(pr_fpr)
        self.pr_tprs.append(pr_tpr)
        self.pr_erpos_fprs.append(pr_erpos_fpr)
        self.pr_erpos_tprs.append(pr_erpos_tpr)

    def _close_roc_plots(self):
        for title, fig, ax, fprs, tprs, aucs in (
            (
                "ROC(ER)",
                self.er_roc_curve_fig,
                self.er_roc_curve_ax,
                self.er_fprs,
                self.er_tprs,
                self.er_auc_rocs,
            ),
            (
                "ROC(ER|PR-)",
                self.er_prneg_roc_curve_fig,
                self.er_prneg_roc_curve_ax,
                self.er_prneg_fprs,
                self.er_prneg_tprs,
                self.er_prneg_auc_rocs,
            ),
            (
                "ROC(PR)",
                self.pr_roc_curve_fig,
                self.pr_roc_curve_ax,
                self.pr_fprs,
                self.pr_tprs,
                self.pr_auc_rocs,
            ),
            (
                "ROC(PR|ER+)",
                self.pr_erpos_roc_curve_fig,
                self.pr_erpos_roc_curve_ax,
                self.pr_erpos_fprs,
                self.pr_erpos_tprs,
                self.pr_erpos_auc_rocs,
            ),
        ):
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.nanmean(
                [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0
            )
            mean_tpr[-1] = 1.0
            mean_auc = np.nanmean(aucs)
            std_auc = np.nanstd(aucs)
            ax.plot(
                mean_fpr,
                mean_tpr,
                color="black",
                label=f"Mean (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
            )

            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(title)
            ax.legend(loc="lower right")
            ax.grid()
            fig.tight_layout()
            fig.savefig(f"{self.file.name.split('.')[0]}_{title}.png")
            plt.close(fig)

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

        self._scatter_plots(
            list(zip(y_pred_er, y_pred_pr)),
            list(zip(y_true_er, y_true_pr)),
        )

        self._plot_rocs(y_true, y_true_er, y_pred_er, y_true_pr, y_pred_pr)

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
        self._close_roc_plots()

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
