import json
import os

from lxml import etree

os.chdir("../../data/")

with open("metadata.repository.2024-10-11.json") as f:
	metadata = json.load(f)

no_pr = 0
pr_neg = 0
pr_pos = 0
pr_indet = 0
no_er = 0
er_neg = 0
er_pos = 0
er_indet = 0
for file_meta in metadata:
	os.chdir(file_meta["file_id"])
	with open(file_meta["file_name"]) as f:
		patient_data = etree.parse(f).getroot().find("{http://tcga.nci/bcr/xml/clinical/brca/2.7}patient")
	pr = patient_data.find("{http://tcga.nci/bcr/xml/clinical/brca/shared/2.7}breast_carcinoma_progesterone_receptor_status")
	if pr.attrib['procurement_status'] != "Completed":
		no_pr += 1
	elif pr.text == "Negative":
		pr_neg += 1
	elif pr.text == "Positive":
		pr_pos += 1
	elif pr.text == "Indeterminate":
		pr_indet += 1
	else:
		print(f"Could not parse pr: {pr.text} in {file_meta["file_name"]}")

	er = patient_data.find("{http://tcga.nci/bcr/xml/clinical/brca/shared/2.7}breast_carcinoma_estrogen_receptor_status")
	if er.attrib['procurement_status'] != "Completed":
		no_er += 1
	elif er.text == "Negative":
		er_neg += 1
	elif er.text == "Positive":
		er_pos += 1
	elif er.text == "Indeterminate":
		er_indet += 1
	else:
		print(f"Could not parse er: {er.text} in {file_meta["file_name"]}")

	os.chdir("..")

print(f"no_pr: {no_pr}, pr-: {pr_neg}, pr_indet: {pr_indet}, pr+: {pr_pos}, no_er: {no_er}, er-: {er_neg}, er_indet: {er_indet}, er+: {er_pos}")