import json
import os
from typing import Any  # noqa: F401

from collections import Counter
from lxml import etree  # type: ignore
from matplotlib import pyplot as plt

os.chdir("../../data/")

with open("metadata.repository.2024-10-11.json") as f:
    metadata = json.load(f)

tss_pr_pos: dict[str, int] = Counter()
tss_pr_neg: dict[str, int] = Counter()
tss_er_pos: dict[str, int] = Counter()
tss_er_neg: dict[str, int] = Counter()
no_graph = 0
num_cases = 0
for file_meta in metadata:
    # All the associated entities are length 1 for these files
    case_id = file_meta["associated_entities"][0]["case_id"]

    with open(file_meta["file_id"] + "/" + file_meta["file_name"]) as f:
        patient_data = (
            etree.parse(f)
            .getroot()
            .find("{http://tcga.nci/bcr/xml/clinical/brca/2.7}patient")
        )  # type: Any

    barcode = patient_data.find(
        "{http://tcga.nci/bcr/xml/shared/2.7}bcr_patient_barcode"
    ).text
    if not os.path.isfile(f"graphs/{barcode}_ShuffleNet.pkl"):
        no_graph += 1
        continue

    pr = patient_data.find(
        "{http://tcga.nci/bcr/xml/clinical/brca/shared/2.7}breast_carcinoma_progesterone_receptor_status"
    )  # type: Any
    er = patient_data.find(
        "{http://tcga.nci/bcr/xml/clinical/brca/shared/2.7}breast_carcinoma_estrogen_receptor_status"
    )  # type: Any

    if (
        pr.attrib["procurement_status"] != "Completed"
        or er.attrib["procurement_status"] != "Completed"
        or pr.text == "Indeterminate"
        or er.text == "Indeterminate"
    ):
        continue

    tss = patient_data.find(
        "{http://tcga.nci/bcr/xml/shared/2.7}tissue_source_site"
    ).text

    if pr.text == "Negative":
        tss_pr_neg[tss] += 1
    elif pr.text == "Positive":
        tss_pr_pos[tss] += 1
    else:
        raise RuntimeError(f"Could not parse pr: {pr.text} in {file_meta['file_name']}")

    if er.text == "Negative":
        tss_er_neg[tss] += 1
    elif er.text == "Positive":
        tss_er_pos[tss] += 1
    else:
        raise RuntimeError(f"Could not parse er: {er.text} in {file_meta['file_name']}")

    num_cases += 1

print("No graph: ", no_graph)
print("Valid: ", num_cases)
print(
    "PR+:",
    sum(tss_pr_pos.values()),
    "PR-:",
    sum(tss_pr_neg.values()),
    "ER+:",
    sum(tss_er_pos.values()),
    "ER-:",
    sum(tss_er_neg.values()),
)

fig, ax = plt.subplots()
all_centres = list(
    set(tss_pr_pos.keys())
    .union(set(tss_pr_neg.keys()))
    .union(set(tss_er_pos.keys()))
    .union(set(tss_er_neg.keys()))
)
BAR_WIDTH = 0.35

for i, c in enumerate(all_centres):
    p1 = ax.bar(i, tss_pr_pos[c], label="PR+", width=BAR_WIDTH, color="blue")
    p2 = ax.bar(
        i,
        tss_pr_neg[c],
        label="PR-",
        bottom=tss_pr_pos[c],
        width=BAR_WIDTH,
        color="orange",
    )
    p3 = ax.bar(
        i + BAR_WIDTH, tss_er_pos[c], label="ER+", width=BAR_WIDTH, color="cyan"
    )
    p4 = ax.bar(
        i + BAR_WIDTH,
        tss_er_neg[c],
        bottom=tss_er_pos[c],
        label="ER-",
        width=BAR_WIDTH,
        color="yellow",
    )

plt.xticks([i + BAR_WIDTH / 2 for i in range(len(all_centres))], all_centres)
plt.legend([p1, p2, p3, p4], ["PR+", "PR-", "ER+", "ER-"])
plt.show()
