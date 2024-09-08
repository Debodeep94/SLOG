import pandas as pd
import json
import numpy as np
from modules.utils import convert
# import the csv

chex = pd.read_csv('/home/debodeep.banerjee/chexpert-labeler/train_only_find0.0.csv')
reports = list(chex['Reports'])
chex = chex.iloc[:,1:]
chex_cols=list(chex.columns)
chex[chex_cols] = np.vectorize(convert)(chex[chex_cols])
chex['info_score']=chex[chex_cols].sum(axis=1)
info_score = list(chex['info_score'])
# import the image ids

with open(r"/home/debodeep.banerjee/R2Gen/data/mimic/only_find_new_split.json", 'r') as f:
    # Load the contents of the file into a variable
    data = json.load(f)
image_id=[]
for i in data['train']:
    image_id.append(i['study_id'])

data = {'study_id': image_id, 'report': report, 'Column3': info_score}
df = pd.DataFrame(data)