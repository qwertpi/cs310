import json
import os

from typing import Optional
from lxml import etree

os.chdir("../../data/")

with open("metadata.repository.2024-10-11.json") as f:
	metadata = json.load(f)

cases: dict[str, list[Optional[str]]] = {}
for file_meta in metadata:
	os.chdir(file_meta["file_id"])
	ae = file_meta['associated_entities']
	if len(ae) != 1:
		print(f"ae = {ae} not len 1")
	case_id = ae[0]["case_id"]
	if cases.get(case_id) is None:
		cases[case_id] = [None, None]
	else:
		print(f"Case {case_id} already seen")

	with open(file_meta["file_name"]) as f:
		patient_data = etree.parse(f).getroot().find("{http://tcga.nci/bcr/xml/clinical/brca/2.7}patient")

	pr = patient_data.find("{http://tcga.nci/bcr/xml/clinical/brca/shared/2.7}breast_carcinoma_progesterone_receptor_status")
	if pr.attrib['procurement_status'] != "Completed":
		cases[case_id][0] = "A"
	elif pr.text == "Negative":
		cases[case_id][0] = "N"
	elif pr.text == "Positive":
		cases[case_id][0] = "P"
	elif pr.text == "Indeterminate":
		cases[case_id][0] = "U"
	else:
		print(f"Could not parse pr: {pr.text} in {file_meta["file_name"]}")

	er = patient_data.find("{http://tcga.nci/bcr/xml/clinical/brca/shared/2.7}breast_carcinoma_estrogen_receptor_status")
	if er.attrib['procurement_status'] != "Completed":
		cases[case_id][1] = "A"
	elif er.text == "Negative":
		cases[case_id][1] = "N"
	elif er.text == "Positive":
		cases[case_id][1] = "P"
	elif er.text == "Indeterminate":
		cases[case_id][1] = "U"
	else:
		print(f"Could not parse er: {er.text} in {file_meta["file_name"]}")

	os.chdir("..")

results = cases.values()

print(f"no_pr: {sum((1 for t in results if t[0] == "A"))}, pr-: {sum((1 for t in results if t[0] == "N"))}, pr_indet: {sum((1 for t in results if t[0] == "U"))}, pr+: {sum((1 for t in results if t[0] == "P"))}")
print(f"no_er: {sum((1 for t in results if t[1] == "A"))}, er-: {sum((1 for t in results if t[1] == "N"))}, er_indet: {sum((1 for t in results if t[1] == "U"))}, er+: {sum((1 for t in results if t[1] == "P"))}")
print(f"One of pr and er well defined: {sum((1 for t in results if t[0] in ("P", "N") or t[1] in ("P", "N")))}")
print(f"Both of pr and er well defined: {sum((1 for t in results if t[0] in ("P", "N") and t[1] in ("P", "N")))}")