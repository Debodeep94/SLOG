
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import torch
import openpyxl
from utils import *

pred_data="./Documents/r2gen/chexpert_labeler/rules/111.0.csv"
image_id_path=r"./Documents/r2gen/chexpert_labeler/chexpert_eval/vanilla/img_ids_111.0.pt"

prec_dict, recall_dict, f1_dict=ce_metrics(pred_data,image_id_path, 1,mode='impression', gt_data=None)
print(prec_dict)
print(recall_dict)
print(f1_dict)

