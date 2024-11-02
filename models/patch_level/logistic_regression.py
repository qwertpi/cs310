import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from PatchModelTrainer import PatchModelTrainer

trainer = PatchModelTrainer()

MEAN_AGGREGATOR = lambda p: np.argmax(np.mean(p, axis=0))  # noqa: E731
VOTE_AGGREGATOR = lambda p: round(np.mean(np.argmax(p, axis=1)))  # noqa: E731
for aggregator_name, aggregator in [
    ("mean", MEAN_AGGREGATOR),
    ("vote", VOTE_AGGREGATOR),
]:
    for model_name, label_idx in (
        (f"ER_{aggregator_name}", 0),
        (f"PR_{aggregator_name}", 1),
    ):
        trainer.train_and_validate(
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    solver="newton-cholesky",
                    max_iter=50000,
                    n_jobs=-1,
                ),
            ),
            aggregator,
            model_name,
            label_idx,
        )
