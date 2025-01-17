from functools import partial
import sys

sys.path.insert(0, "..")

import click
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore

from PatchModelTrainer import PatchModelTrainer  # type: ignore


@click.command()
@click.argument("c", type=float)
def train_model(c: float):
    model_func = partial(
        LogisticRegression,
        penalty="l2",
        C=c,
        class_weight="balanced",
        solver="newton-cholesky",
        max_iter=50000,
        n_jobs=-1,
    )
    PatchModelTrainer().train_and_validate(
        lambda: make_pipeline(model_func()),
        lambda: make_pipeline(model_func()),
        f"logreg_c={c}",
    )


if __name__ == "__main__":
    train_model()
