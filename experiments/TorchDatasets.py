from __future__ import annotations
from abc import ABC, abstractmethod
import pickle
from random import shuffle
from typing import final

import torch

from torch_geometric.data import Data  # type: ignore


class Dataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __init__(self):
        self.lens: list[int]

    @final
    def __len__(self):
        return sum(self.lens)

    @final
    def get_folds(self):
        folds: list[tuple[list[int], list[int]]] = []
        for i in range(0, 5):
            train_idxs: list[int] = []
            val_idxs: list[int] = []
            for j in range(0, 5):
                fold_idxs = list(range(sum(self.lens[:j]), sum(self.lens[: j + 1])))
                if i == j:
                    val_idxs = fold_idxs
                else:
                    shuffle(fold_idxs)
                    train_idxs.extend(fold_idxs)
            folds.append((train_idxs, val_idxs))
        return folds

    @staticmethod
    @abstractmethod
    def get_feat_dim() -> int:
        pass

    @staticmethod
    @abstractmethod
    def get_test_data() -> list[Data]:
        pass


class TCGADataset(Dataset):
    def __init__(self):
        self.graphs: list[Data] = []
        self.lens = []
        for i in range(0, 5):
            with open(f"../../../data/train_data_fold_{i}.pkl", "rb") as f:
                gs = [t[1] for t in pickle.load(f)]
                self.lens.append(len(gs))
                self.graphs.extend(gs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    @staticmethod
    def get_test_data():
        with open("../../../data/test_data.pkl", "rb") as f:
            return [t[1] for t in pickle.load(f)]

    @staticmethod
    def get_feat_dim():
        return 1024


class ABCTBDataset(Dataset):
    def __init__(self):
        self.PREFIX = "/dcs/large/u2220772/data_train_"
        self.lens: list[int] = []
        self.cached_data: list[Data] = []
        self.cached_path: str = ""
        for i in range(0, 5):
            with open(f"{self.PREFIX}fold{i}.pkl", "rb") as f:
                self.lens.append(len(pickle.load(f)))

    def __getitem__(self, idx):
        acc = self.lens[0]
        i = 0
        while acc < idx + 1:
            i += 1
            acc += self.lens[i]
        acc -= self.lens[i]

        target_file = f"{self.PREFIX}fold{i}.pkl"
        if self.cached_path != target_file:
            with open(target_file, "rb") as f:
                del self.cached_data
                self.cached_path = target_file
                self.cached_data = pickle.load(f)
        graph = self.cached_data[idx - acc]
        return graph

    @staticmethod
    def get_test_data():
        with open("/dcs/large/u2220772/data_test.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def get_feat_dim():
        return 1040
