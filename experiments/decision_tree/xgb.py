import sys

sys.path.insert(0, "..")

import numpy as np
from sklearn.pipeline import make_pipeline  # type: ignore
from xgboost import XGBClassifier, XGBRFClassifier

from PatchModelTrainer import PatchModelTrainer  # type: ignore

MEAN_AGGREGATOR = lambda p: int(np.argmax(np.mean(p, axis=0)))  # noqa: E731

NUM_POS = {"ER": 494896, "PR": 420617}
NUM_EXAMPLES = 646152

trainer = PatchModelTrainer()


def train_model():
    for model in (XGBClassifier, XGBRFClassifier):
        for receptor in ("ER", "PR"):
            class_imbalance = (NUM_EXAMPLES - NUM_POS[receptor]) / NUM_POS[receptor]
            for gamma in (0, 1, 100):
                for l1 in (0, 1, 100):
                    for l2 in (0, 1, 100):
                        trainer.train_and_validate(
                            lambda: make_pipeline(
                                model(
                                    device="gpu",
                                    gamma=gamma,
                                    reg_alpha=l1,
                                    reg_lambda=l2,
                                    scale_pos_weight=class_imbalance,
                                ),
                            ),
                            MEAN_AGGREGATOR,
                            f"{model.__name__}_{receptor}_gamma={gamma}_l1={l1}_l2={l2}",
                            {"ER": 0, "PR": 1}[receptor],
                        )


if __name__ == "__main__":
    train_model()
