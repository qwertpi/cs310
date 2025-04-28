import pickle
import torch
from tqdm import tqdm

dataset = None
for dataset_path in tqdm(
    [f"/dcs/large/u2220772/data_train_fold{i}.pkl" for i in range(0, 5)] +
    [f"/dcs/large/u2220772/data_test.pkl"]
):
    print(dataset_path)
    del dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    ys = [(graph.y[0][0].item(), graph.y[0][1].item()) for graph in dataset]
    print(len([1 for e, p in ys if e and p]))
    print(len([1 for e, p in ys if e and not p]))
    print(len([1 for e, p in ys if not e and p]))
    print(len([1 for e, p in ys if not e and not p]))
print("Done")

