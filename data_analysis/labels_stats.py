import pickle

import torch_geometric.data  # type: ignore

with open("../../data/train_data.pkl", "rb") as f:
    train_set: list[tuple[str, torch_geometric.data.Data, tuple[bool, bool]]] = (
        pickle.load(f)
    )
with open("../../data/test_data.pkl", "rb") as f:
    test_set: list[tuple[str, torch_geometric.data.Data, tuple[bool, bool]]] = (
        pickle.load(f)
    )

for dataset, dataset_name in ((test_set, "test"), (train_set, "train")):
    print(dataset_name)
    labels = [ls for _, _, ls in dataset]
    print(f"ER+/PR+: {sum((1 for er, pr in labels if er == 1 and pr == 1))}")
    print(f"ER+/PR-: {sum((1 for er, pr in labels if er == 1 and pr == 0))}")
    print(f"ER-/PR+: {sum((1 for er, pr in labels if er == 0 and pr == 1))}")
    print(f"ER-/PR-: {sum((1 for er, pr in labels if er == 0 and pr == 0))}")
