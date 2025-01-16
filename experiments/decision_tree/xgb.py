from functools import partial
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
        er_class_imbalance = (NUM_EXAMPLES - NUM_POS["ER"]) / NUM_POS["ER"]
        pr_class_imbalance = (NUM_EXAMPLES - NUM_POS["PR"]) / NUM_POS["PR"]
        for gamma in (0, 1, 100):
            for l1 in (0, 1, 100):
                for l2 in (0, 1, 100):
                    model_func = partial(
                        model, device="gpu", gamma=gamma, reg_alpha=l1, reg_lambda=l2
                    )
                    trainer.train_and_validate(
                        lambda: make_pipeline(
                            model_func(scale_pos_weight=er_class_imbalance)
                        ),
                        lambda: make_pipeline(
                            model_func(scale_pos_weight=pr_class_imbalance)
                        ),
                        f"{model.__name__}_gamma={gamma}_l1={l1}_l2={l2}",
                    )


if __name__ == "__main__":
    train_model()
