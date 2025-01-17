from functools import partial
import sys

sys.path.insert(0, "..")

from sklearn.pipeline import make_pipeline  # type: ignore
from tqdm import tqdm
from xgboost import XGBClassifier

from PatchModelTrainer import PatchModelTrainer  # type: ignore


NUM_POS = {"ER": 494896, "PR": 420617}
NUM_EXAMPLES = 646152

trainer = PatchModelTrainer()


def train_model():
    er_class_imbalance = (NUM_EXAMPLES - NUM_POS["ER"]) / NUM_POS["ER"]
    pr_class_imbalance = (NUM_EXAMPLES - NUM_POS["PR"]) / NUM_POS["PR"]
    for gamma in tqdm([0, 1, 10, 100]):
        l1 = 0
        l2 = 0
        model_func = partial(XGBClassifier, device="gpu", gamma=gamma, reg_alpha=l1, reg_lambda=l2)
        trainer.train_and_validate(
            lambda: make_pipeline(
                model_func(scale_pos_weight=er_class_imbalance)
            ),
            lambda: make_pipeline(
                model_func(scale_pos_weight=pr_class_imbalance)
            ),
            f"xgb_g{gamma}_l1{l1}_l2{l2}",
        )


if __name__ == "__main__":
    train_model()
