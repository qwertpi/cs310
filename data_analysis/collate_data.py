import json
import os
import pickle

from lxml import etree  # type: ignore
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from tqdm import tqdm

os.chdir("../../data/")

with open("metadata.repository.2024-10-11.json") as f:
    metadata = json.load(f)

# (tissue_source, barcode, graph, (er, pr))
data: list[tuple[str, str, Data, tuple[bool, bool]]] = []

for file_meta in tqdm(metadata):
    # All the associated entities are length 1 for these files
    case_id = file_meta["associated_entities"][0]["case_id"]

    with open(file_meta["file_id"] + "/" + file_meta["file_name"]) as f:
        patient_data = (
            etree.parse(f)
            .getroot()
            .find("{http://tcga.nci/bcr/xml/clinical/brca/2.7}patient")
        )

    barcode = patient_data.find(
        "{http://tcga.nci/bcr/xml/shared/2.7}bcr_patient_barcode"
    ).text

    try:
        with open(f"graphs/{barcode}_ShuffleNet.pkl", "rb") as f:
            graph: Data = pickle.load(f)
    except FileNotFoundError:
        continue

    pr = patient_data.find(
        "{http://tcga.nci/bcr/xml/clinical/brca/shared/2.7}breast_carcinoma_progesterone_receptor_status"
    )
    er = patient_data.find(
        "{http://tcga.nci/bcr/xml/clinical/brca/shared/2.7}breast_carcinoma_estrogen_receptor_status"
    )

    if (
        pr.attrib["procurement_status"] != "Completed"
        or er.attrib["procurement_status"] != "Completed"
        or pr.text == "Indeterminate"
        or er.text == "Indeterminate"
    ):
        continue

    tissue_source: str = patient_data.find(
        "{http://tcga.nci/bcr/xml/shared/2.7}tissue_source_site"
    ).text

    if pr.text == "Negative":
        pr_pos = False
    elif pr.text == "Positive":
        pr_pos = True
    else:
        raise RuntimeError(f"Could not parse pr: {pr.text} in {file_meta['file_name']}")

    if er.text == "Negative":
        er_pos = False
    elif er.text == "Positive":
        er_pos = True
    else:
        raise RuntimeError(f"Could not parse er: {er.text} in {file_meta['file_name']}")
    data.append((tissue_source, barcode, graph, (er_pos, pr_pos)))

if len(set(t[1] for t in data)) != len(data):
    raise RuntimeError("Number of samples not equal to number of patients")

# The imbalance of the number of samples between centres means 5 fold gives ~11% test data, whereas 4 fold gives ~23%
splitter = StratifiedGroupKFold(n_splits=4, shuffle=True)
groups: list[str]
x: list[Data]
y: list[tuple[int, int]]
groups, _, x, y = zip(*data)  # type: ignore
y_compact = [str(a) + str(b) for a, b in y]
train_idxs, test_idxs = next(splitter.split(x, y_compact, groups))

train_set = [data[i][1:] for i in train_idxs]
test_set = [data[i][1:] for i in test_idxs]

scaler = StandardScaler()
scaler.fit(torch.cat([t[1].x for t in train_set]))

for _, graph, (_, _) in train_set:
    graph.x = torch.tensor(scaler.transform(graph.x), dtype=torch.float32)

for _, graph, (_, _) in test_set:
    graph.x = torch.tensor(scaler.transform(graph.x), dtype=torch.float32)

print(len(data))
print(len(train_set))
print(len(test_set))
print(set([data[i][0] for i in train_idxs]))
print(set([data[i][0] for i in test_idxs]))
with open("train_data.pkl", "wb") as f:
    pickle.dump(train_set, f)
with open("test_data.pkl", "wb") as f:
    pickle.dump(test_set, f)
