import pandas as pd
import json
import csv

#csv_file_path = "/home/debodeep.banerjee/R2Gen/csv_outputs/only_find/alternative/alter_R2Gen"+str(self.surr_weight)+".csv"

# Open json file
with open ('/home/debodeep.banerjee/R2Gen/data/mimic/imp_n_find_split.json') as f:
    data = json.load(f)

# Store impression
impression=[]
for i in data:
    print(i)
    for j in data[i]:
        impression.append(j['impression'])
print(len(impression))

# Store findings
findings=[]
for i in data:
    print(i)
    for j in data[i]:
        findings.append(j['findings'])
print(len(findings))


# Open the CSV file in write mode

csv_file_path_imp = "/home/debodeep.banerjee/R2Gen/csv_outputs/imp_n_find/impressions.csv"

# Open the CSV file in write mode
with open(csv_file_path_imp, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write each string in the list as a row in the CSV file
    for string in impression:
        csv_writer.writerow([string])
        
csv_file_path_find = "/home/debodeep.banerjee/R2Gen/csv_outputs/imp_n_find/findings.csv"

# Open the CSV file in write mode
with open(csv_file_path_find, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write each string in the list as a row in the CSV file
    for string in findings:
        csv_writer.writerow([string])
        