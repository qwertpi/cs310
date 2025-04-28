import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
from tqdm import tqdm
import torch

graphs = []
patient_ids = []
labels: list[tuple[bool, bool]] = []
for filename in tqdm(list(Path(".").glob("*.pth"))):
    graph = torch.load(filename)
    patient_ids.append(graph.patient)
    labels.append((graph.y["ER Result"].item(), graph.y["PR Result"].item()))
    graph.y = torch.tensor(
        (graph.y["ER Result"].item(), graph.y["PR Result"].item()), dtype=torch.float
    ).unsqueeze(0)
    del graph["feat_names"]
    del graph["coords"]
    del graph["id"]
    del graph["patient"]
    graphs.append(graph)

with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True)
labels_compact = [str(a) + str(b) for a, b in labels]
train_idxs, test_idxs = next(splitter.split(graphs, labels_compact, patient_ids))

scaler = StandardScaler()
scaler.fit(torch.cat([graphs[i].x for i in train_idxs]))
for graph in tqdm(graphs):
    graph.x = torch.tensor(scaler.transform(graph.x), dtype=torch.float32)

with open("data_test.pkl", "wb") as f:
    pickle.dump([graphs[i] for i in test_idxs], f)

graphs = [graphs[i] for i in train_idxs]
labels_compact = [labels_compact[i] for i in train_idxs]
patient_ids = [patient_ids[i] for i in train_idxs]
folds = []
for _, idxs in splitter.split(graphs, labels_compact, patient_ids):
    folds.append([graphs[i] for i in idxs])
del graphs

for i, fold in enumerate(folds):
    with open(f"data_train_fold{i}.pkl", "wb") as f:
        pickle.dump(fold, f)

print("Done")
