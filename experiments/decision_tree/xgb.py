from functools import partial
import sys

sys.path.insert(0, "..")

from sklearn.pipeline import make_pipeline  # type: ignore
from xgboost import XGBClassifier

from PatchModelTrainer import PatchModelTrainer  # type: ignore


NUM_POS = {"ER": 494896, "PR": 420617}
NUM_EXAMPLES = 646152

trainer = PatchModelTrainer()


def train_model():
    er_class_imbalance = (NUM_EXAMPLES - NUM_POS["ER"]) / NUM_POS["ER"]
    pr_class_imbalance = (NUM_EXAMPLES - NUM_POS["PR"]) / NUM_POS["PR"]
    model_func = partial(XGBClassifier, device="gpu")
    trainer.train_and_validate(
        lambda: make_pipeline(model_func(scale_pos_weight=er_class_imbalance)),
        lambda: make_pipeline(model_func(scale_pos_weight=pr_class_imbalance)),
        "xgb",
    )


if __name__ == "__main__":
    train_model()
