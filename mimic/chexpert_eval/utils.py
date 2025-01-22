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

def ce_metrics(pred_data,image_id_path, mention,mode='impression', gt_data=None, ind_samp = None):
    #load pred data
    pred_df=pd.read_csv(pred_data)
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
    # load gt data
    if mode=='impression':
        gt_df = pd.read_excel(r"./Documents/r2gen/chexpert_labeler/mimic-cxr-2.0.0-chexpert.xlsx")    
        gt_df['study_id'] = ['s'+str(i) for i in gt_df['study_id']]
        test_ids=torch.load(image_id_path)
        #test_study_ids
        gt_df=gt_df[gt_df['study_id'].isin(test_ids)]
    else:
        gt_df = pd.read_csv(gt_data)
    
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


def draw_samples(data, num_samples):

        labels = data.columns[1:]
        data[labels] = np.vectorize(convert)(data[labels])
        label_set = np.array(data[labels])
        examples = list(data['study_id'])
        example_dict=dict(zip(examples, label_set))
        bucket_label_0=[]
        bucket_label_1=[]
        bucket_label_2=[]
        for i in examples:
            if 1 in example_dict[i]:
                bucket_label_1.append(i)
            if 2 in example_dict[i]:
                bucket_label_2.append(i)
            if 0 in example_dict[i]:
                bucket_label_0.append(i)
            else:
                print('number outside expectation')
                print(i)
                print(example_dict[i])
            # pick one sample from bucket 0
        # check for the least occurrance of label 
        # Sample from that bucket
        # Repeat till we meet the required number

        sample = []
        count=0
        while count < num_samples:
            draw = random.sample(bucket_label_0,1)
            #print(draw)
            sample.append(draw)
            try:
                bucket_label_0.remove(draw[0])
            except:
                print('id not available in bucket_label_0')
            try:
                bucket_label_1.remove(draw[0])
            except:
                print('id not available in bucket_label_1')
            try:
                bucket_label_2.remove(draw[0])
            except:
                print('id not available in bucket_label_2')
            count = count+1
            # Find unique elements and their counts
            unique_elements, counts = np.unique(example_dict[draw[0]], return_counts=True)

            # Find the index of the least frequent number
            least_frequent_index = np.argmin(counts)

            # Get the least frequent number
            minority_label = unique_elements[least_frequent_index]
            print('lest frequent label: ', minority_label)
            if minority_label==0:
                draw = random.sample(bucket_label_0,1)
                print('drawing from bucket: bucket_label_0')
            if minority_label==1:
                print('drawing from bucket: bucket_label_1')
                draw = random.sample(bucket_label_1,1)
            if minority_label==2:
                print('drawing from bucket: bucket_label_2')
                draw = random.sample(bucket_label_2,1)
            print('=====================================')
            #print(count)
            #print(first_id)
        return sample


def sample_check(pred_data,image_id_path, mention,mode='impression', gt_data=None):
    pred_df=pd.read_csv(pred_data)
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
    # load gt data
    test_ids=torch.load(image_id_path)
    if mode=='impression':
        gt_df = pd.read_excel(r"./Documents/r2gen/chexpert_labeler/mimic-cxr-2.0.0-chexpert.xlsx")    
        gt_df['study_id'] = ['s'+str(i) for i in gt_df['study_id']]
        
        #test_study_ids
        gt_df=gt_df[gt_df['study_id'].isin(test_ids)]
    else:
        gt_df = pd.read_csv(gt_data)
    
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
    
    y_pred = np.array(pred_df)
    match_box=[]
    for i in range((y_pred.shape[0])):
        for j in range(len(y_true[i])):
            if y_pred[i][j]+y_true[i][j]==2:
                match_box.append(i)
    d = {x:match_box.count(x) for x in match_box}
    for k in list(d.keys()):
        if d[k]==4:
            print(f'the key is: {k}')
            print(f'the val is: {d[k]}')

    print(y_pred[1244])
    print(y_true[1244])
    print(test_ids[1244])
    return match_box
