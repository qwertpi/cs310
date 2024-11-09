import click
import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from PatchModelTrainer import PatchModelTrainer

MEAN_AGGREGATOR = lambda p: np.argmax(np.mean(p, axis=0))  # noqa: E731
VOTE_AGGREGATOR = lambda p: round(np.mean(np.argmax(p, axis=1)))  # noqa: E731

# for aggregator_name, aggregator in [
#     ("mean", MEAN_AGGREGATOR),
#     ("vote", VOTE_AGGREGATOR),
# ]:
#     for model_name, label_idx in (
#         (f"ER_{aggregator_name}", 0),
#         (f"PR_{aggregator_name}", 1),
#     ):


@click.command()
@click.argument("receptor")
@click.argument("aggregator")
@click.argument("l1ratio", type=float)
@click.argument("c", type=float)
def train_model(receptor: str, aggregator: str, l1ratio: float, c: float):
    PatchModelTrainer().train_and_validate(
        lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                C=c,
                class_weight="balanced",
                solver="saga",
                max_iter=50000,
                n_jobs=-1,
                l1_ratio=l1ratio,
            ),
        ),
        {"mean": MEAN_AGGREGATOR, "vote": VOTE_AGGREGATOR}[aggregator],
        f"{receptor}_{aggregator}_{l1ratio}_{c}",
        {"ER": 0, "PR": 1}[receptor],
    )


if __name__ == "__main__":
    train_model()
