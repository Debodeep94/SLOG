import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import torch

#in the orginal dataset, 1: present; 0: not present; -1: ambigous; Nan: Missing. We make the following changes
# 1:1; 0:-1; -1:0; missing:0
def convert(val):
    if val==-1:
        return  2# This is just for matching the chexpert with ground truth
    elif val == 1:
        return val
    else:
        return 0
def ambi(val):
    if val == -1:
        return 1
    else:
        return 0
    
def positive(val):
    if val == 1:
        return 1
    else:
        return 0
    
def negative(val):
    if val == 0.0:
        return 1
    else:
        return 0



def compute_mlc_f1(gt, pred, label_set):
    res_mlc = {}
    score = 0
    for i, label in enumerate(label_set):
        res_mlc['F1_' + label] = round(f1_score(gt[:, i], pred[:, i], zero_division=0)*100,2)
        score += res_mlc['F1_' + label]
        #print(f'{i}:{}')
    res_mlc['AVG_F1'] = score / len(label_set)

    res_mlc['F1_MACRO'] = round(f1_score(gt, pred, average="macro", zero_division=0)*100,2)
    res_mlc['F1_MICRO'] = round(f1_score(gt, pred, average="micro", zero_division=0)*100,2)
    #res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro", zero_division=0)
    #res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro", zero_division=0)
    #res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro", zero_division=0)
    #res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro", zero_division=0)

    return res_mlc

#compute_mlc_f1(y_true,y_pred, gt_cols)
def compute_mlc_recall(true_data, pred, label_set):
    res_mlc = {}
    score = 0
    for i, label in enumerate(label_set):
        res_mlc['RECALL_' + label] = round(recall_score(true_data[:, i], pred[:, i], zero_division=0)*100,2)
        score += res_mlc['RECALL_' + label]
    res_mlc['AVG_RECALL'] = score / len(label_set)

    #res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro", zero_division=0)
    #res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro", zero_division=0)
    res_mlc['RECALL_MACRO'] = round(recall_score(true_data, pred, average="macro", zero_division=0)*100,2)
    res_mlc['RECALL_MICRO'] = round(recall_score(true_data, pred, average="micro", zero_division=0)*100,2)
    #res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro", zero_division=0)
    #res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro", zero_division=0)

    return res_mlc

def compute_mlc_precision(true_data, pred, label_set):
    res_mlc = {}
    score = 0
    for i, label in enumerate(label_set):
        res_mlc['PRECISION_' + label] = round(precision_score(true_data[:, i], pred[:, i], zero_division=0)*100,2)
        score += res_mlc['PRECISION_' + label]
    #res_mlc['AVG_PRECISION'] = score / len(label_set)

    #res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro", zero_division=0)
    #res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro", zero_division=0)
    res_mlc['PRECISION_MACRO'] = round(precision_score(true_data, pred, average="macro", zero_division=0)*100,2)
    res_mlc['PRECISION_MICRO'] = round(precision_score(true_data, pred, average="micro", zero_division=0)*100,2)
    #res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro", zero_division=0)
    #res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro", zero_division=0)

    return res_mlc

def ce_metrics(pred_data_path,gt_data_path,mention=1):
    #load pred data
    pred_df=pd.read_csv(pred_data_path)
    gt_df=pd.read_csv(gt_data_path)
    #print(pred_df)
    pred_df = pred_df.iloc[:,1:] # we don't exclude 'no findings'
    pred_df_cols=list(pred_df.columns)
    #pred_df[pred_df_cols] = np.vectorize(mention)(pred_df[pred_df_cols])
    if mention==1:
        pred_df[pred_df_cols] = np.vectorize(positive)(pred_df[pred_df_cols])
    elif mention == 0:
        pred_df[pred_df_cols] = np.vectorize(negative)(pred_df[pred_df_cols])
    elif mention == 2:
        pred_df[pred_df_cols] = np.vectorize(ambi)(pred_df[pred_df_cols])
    elif mention == 3:
        pred_df[pred_df_cols] = np.vectorize(convert)(pred_df[pred_df_cols])
    filtered_gt = gt_df[pred_df_cols]
    gt_cols=list(filtered_gt.columns)
    if mention==1:
        filtered_gt[gt_cols] = np.vectorize(positive)(filtered_gt[gt_cols])
    elif mention == 0:
        filtered_gt[gt_cols] = np.vectorize(negative)(filtered_gt[gt_cols])
    elif mention == 2:
        filtered_gt[gt_cols] = np.vectorize(ambi)(filtered_gt[gt_cols])
    elif mention == 3:
        filtered_gt[gt_cols] = np.vectorize(convert)(filtered_gt[gt_cols])
    y_true = np.array(filtered_gt)
    print(y_true)
    y_pred = np.array(pred_df)
    #prediction
    prec_dict = compute_mlc_precision(y_true,y_pred, gt_cols)
    f1_dict = compute_mlc_f1(y_true,y_pred, gt_cols)
    recall_dict = compute_mlc_recall(y_true,y_pred, gt_cols)
    for i in range(y_true.shape[1]):
        tp=0
        true_val = y_true[:,i]
        pred_val=y_pred[:,i]
        #tn, fp, fn, tp = sklearn.metrics.confusion_matrix(gt, pred).ravel()
        tp = ((true_val == 1) & (pred_val == 1)).sum().item()
        fp = ((true_val == 0) & (pred_val == 1)).sum().item()
        tn = ((true_val == 0) & (pred_val == 0)).sum().item()
        fn = ((true_val == 1) & (pred_val == 0)).sum().item()
        print(gt_cols[i])
        print(f'tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}')
    return prec_dict, recall_dict, f1_dict


