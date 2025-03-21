from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, ParamSpec, TypeVar

from numpy.typing import NDArray

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat

from sklearn.metrics import (  # type: ignore
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ABCTB stats
ER_POS_PROB = 2001 / 2538
ER_POS_GIVEN_PR_POS_PROB = 1743 / 1792
ER_POS_GIVEN_PR_NEG_PROB = 258 / 746
PR_POS_PROB = 1792 / 2538
PR_POS_GIVEN_ER_POS_PROB = 1743 / 2001
PR_POS_GIVEN_ER_NEG_PROB = 49 / 537

P = ParamSpec("P")


def float_wrapper(f: Callable[P, ConvertibleToFloat]):
    def ret_func(*args: P.args, **kwargs: P.kwargs):
        return float(f(*args, **kwargs))

    return ret_func


T = TypeVar("T")
U = TypeVar("U")


def condition_wrapper(
    f: Callable[[list[int], list[float]], T],
    filter: Callable[[U], bool],
):
    def ret_func(labels: list[U], true: list[int], pred: list[float]):
        idxs = [i for i, l in enumerate(labels) if filter(l)]
        return f(
            [true[i] for i in idxs],
            [pred[i] for i in idxs],
        )

    return ret_func


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


def typed_roc_curve(
    true: list[int], pred: list[float]
) -> tuple[NDArray, NDArray, NDArray]:
    return roc_curve(true, pred)  # type: ignore


def typed_auc_roc_weighted(
    x1: list[int], x2: list[float], weights: Optional[list[float]]
) -> float:
    if len(set(x1)) < 2:
        return float("nan")
    return float_wrapper(roc_auc_score)(x1, x2, sample_weight=weights)  # type: ignore


def typed_auc_roc(x1: list[int], x2: list[float]):
    return typed_auc_roc_weighted(x1, x2, None)


def balanced_auc_roc(
    y_true: list[int],
    y_score: list[float],
    z_true: list[int],
    z_score: list[float],
    scale_pos: float,
):
    weights = [{1: 1 / scale_pos, 0: 1 / (1 - scale_pos)}[l] for l in z_true]
    return typed_auc_roc_weighted(y_true, y_score, weights)


def fstify2a(f):
    def inner(x1, y1, x2, y2, **kwargs):
        return f(x1, y1, **kwargs)

    return inner


def sndify2a(f):
    def inner(x1, y1, x2, y2, **kwargs):
        return f(x2, y2, **kwargs)

    return inner


def flip2a(f):
    def inner(x1, y1, x2, y2, **kwargs):
        return f(x2, y2, x1, y1, **kwargs)

    return inner


def erpr_confusion_matrix(
    er_true: list[int], er_pred: list[float], pr_true: list[int], pr_pred: list[float]
):
    # The labels are stored as booleans but we want 0, 1 for this
    y_true_concat = [f"{int(er)}{int(pr)}" for er, pr in zip(er_true, pr_true)]
    y_pred_concat = [f"{round(er)}{round(pr)}" for er, pr in zip(er_pred, pr_pred)]
    return confusion_matrix(
        y_true_concat, y_pred_concat, labels=["00", "01", "10", "11"], normalize="true"
    )
