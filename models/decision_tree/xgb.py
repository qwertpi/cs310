import sys

sys.path.insert(0, "..")

import click
import numpy as np
from sklearn.pipeline import make_pipeline  # type: ignore
from xgboost import XGBClassifier, XGBRFClassifier

from PatchModelTrainer import PatchModelTrainer  # type: ignore

MEAN_AGGREGATOR = lambda p: int(np.argmax(np.mean(p, axis=0)))  # noqa: E731

NUM_POS = {"ER": 494896, "PR": 420617}
NUM_EXAMPLES = 646152

trainer = PatchModelTrainer()


@click.command()
def train_model():
    for model in (XGBClassifier, XGBRFClassifier):
        for receptor in ("ER", "PR"):
            class_imbalance = (NUM_EXAMPLES - NUM_POS[receptor]) / NUM_POS[receptor]
            """
            Hyperparams:
            n_estimators - number of trees in forest
            max_depth
            max_leaves
            learning rate - eta of boosting
            gamma - minimum loss reduction required to split a leaf
            reg_alpha - amount of L1 regularisation
            reg_lambda - amount of L2 regularisation
            """
            trainer.train_and_validate(
                lambda: make_pipeline(
                    model(device="gpu", scale_pos_weight=class_imbalance),
                ),
                MEAN_AGGREGATOR,
                f"{model.__name__}_{receptor}_defaults",
                {"ER": 0, "PR": 1}[receptor],
            )


if __name__ == "__main__":
    train_model()
