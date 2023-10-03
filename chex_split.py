# This file is used to add the chexpert split to the annotated data
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import random
# Open the file for reading
with open(r"/home/debodeep.banerjee/R2Gen/data/mimic/surr_only_find_filtered.json", 'r') as f:
    # Load the contents of the file into a variable
    data = json.load(f)

# Calculate the number of items for train1 (10% of the total)
num_ft1 = int(len(data["finetune"]) * 0.1)

# Shuffle the "train" list to randomize the selection
random.shuffle(data["finetune"])

# Split the data into train1 and train2
ft1 = data["finetune"][:num_ft1]
ft2 = data["finetune"][num_ft1:]

# Update the "train" key in the data dictionary
data["finetune"] = ft2  # Rest of the items go to train2

# Add train1 as a new key
data["chexpert"] = ft1
for i in data['chexpert']:
    i['split'] = 'chexpert'
out_file = open(r"/home/debodeep.banerjee/R2Gen/data/mimic/surr_chex_only_find.json", "w")

json.dump(data, out_file, indent = 6)

out_file.close()
print("chexpert:", data["chexpert"])
print("finetune:", data["finetune"])