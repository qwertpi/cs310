from collections import defaultdict
import pickle
from pathlib import Path
from tqdm import tqdm
import torch

labels = defaultdict(list)
for filename in tqdm(list(Path(".").glob("*.pth"))):
    graph = torch.load(filename)
    labels[graph["patient"]].append(
        (graph.y["ER Result"].item(), graph.y["PR Result"]).item()
    )

with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)
print("Done")
