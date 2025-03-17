from __future__ import annotations
from abc import ABC, abstractmethod
import pickle

from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
import torch

from torch_geometric.data import Data  # type: ignore


class Dataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def get_folds(self) -> list[tuple[list[int], list[int]]]:
        pass


class TCGADataset(Dataset):
    def __init__(self):
        with open("../../../data/train_data.pkl", "rb") as f:
            self.data: list[tuple[str, Data, tuple[bool, bool]]] = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, graph, labels = self.data[idx]
        graph.y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)
        return graph

    def get_folds(self):  # type: ignore
        groups: list[str] = []
        y: list[tuple[bool, bool]] = []
        y_compact: list[str] = []
        for group, _, graph_label in self.data:
            y_compact.append(str(int(graph_label[0])) + str(int(graph_label[1])))
            y.append(graph_label)
            groups.append(group)
        return StratifiedGroupKFold(n_splits=5, shuffle=True).split(
            y_compact, y_compact, groups
        )


class ABCTBDataset(Dataset):
    def __init__(self):
        self.PREFIX = "/dcs/large/u2220772/data_train_"
        self.lens: list[int] = []
        for i in range(0, 5):
            with open(f"{self.PREFIX}fold{i}.pkl", "rb") as f:
                self.cached_path = f.name
                self.cached_data = pickle.load(f)
                self.lens.append(len(self.cached_data))

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        acc = self.lens[0]
        i = 0
        while acc < idx + 1:
            i += 1
            acc += self.lens[i]
        acc -= self.lens[i]

        target_file = f"{self.PREFIX}fold{i}.pkl"
        if self.cached_path != target_file:
            print("Cache miss")
            with open(target_file, "rb") as f:
                self.cached_path = target_file
                self.cached_data = pickle.load(f)
        else:
            print("Cache hit")
        return self.cached_data[idx - acc]

    def get_folds(self):
        folds: list[tuple[list[int], list[int]]] = []
        for i in range(0, 5):
            train_idxs: list[int] = []
            val_idxs: list[int] = []
            for j in range(0, 5):
                if i == j:
                    val_idxs = list(range(sum(self.lens[:i]), sum(self.lens[: i + 1])))
                train_idxs.extend(range(sum(self.lens[:j]), sum(self.lens[: j + 1])))
            folds.append((train_idxs, val_idxs))
        return folds
