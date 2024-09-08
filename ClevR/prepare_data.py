import json
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import random
import time
from surrogate_module.surrogate_utils import *
import argparse
import re
from PIL import Image
from tqdm import tqdm
def parse_agrs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--proportion",type=float)
    args = parser.parse_args()
    return args

def main():
    args = parse_agrs()
    prop = args.proportion

    path = '/home/debodeep.banerjee/clevr/R2Gen/surrogate/'
    # seq_dict_gt=torch.load( path+'seq_dict_gt.pt',map_location="cpu")
    # seq_dict_pred=torch.load(path+'seq_dict_pred.pt',map_location="cpu")
    # embed_dict_gt=torch.load(path+'tensor_dict_gt.pt')
    embed_dict_pred=torch.load(path+'tensor_dict.pt')

    # print(embed_dict_pred.keys())
    # print(embed_dict_gt.keys())
    print(torch.cuda.memory_summary())
    # Surrogate formation
    chex = pd.read_csv('/home/debodeep.banerjee/clevr/R2Gen/csv_outputs/annotations/correctness_rules.csv')
    #chex = chex.drop([ 'subject_id'], axis=1) #'No Finding','Support Devices',
    #chex=chex.dropna(how='all')#.reset_index(drop=0)
    indices= list(chex.index)
    with open(r"/home/debodeep.banerjee/synthetic/clevr_data_long.json", 'r') as f:
        # Load the contents of the file into a variable
        data = json.load(f)
    '''image_ids=[]
    for i in data:
        for j in data[i]:
            image_ids.append(j['image_id'])
    filtered_s_id=[image_ids[i] for i in indices]
    chex.insert(0, 'image_id', filtered_s_id)''' # We are commenting this line because we are using the ground truth provided by the chexpert
    #chex['image_id'] = ['s'+str(i) for i in chex['image_id']]
    # train portion of the reports
    image_ids_train=[]
    for i in tqdm(data['train']):
        #for j in data[i]:
        image_ids_train.append(i['image_id'])
    #print('image_ids_train: ', image_ids_train)
    # Further filter for the train data
    chex = chex[chex['image_id'].isin(image_ids_train)]
    #nan counts
    #print('chex: ', chex)
    nan_count = (chex.isna().sum()/chex.shape[0])*100
    print('nan counts: ',nan_count)
    # remove columns with more than 95% nans
    #chex = chex.drop(['Pleural Other','Fracture','Lung Lesion', 'Enlarged Cardiomediastinum'], axis=1)
    # chex=chex.fillna(0)
    print(chex.head())
    num_samples = int(len(embed_dict_pred.keys())* prop/100)
    print(f'samples to be collected: {num_samples}')
    print(list(embed_dict_pred.keys()))
    #embed_dict = torch.load('/home/debodeep.banerjee/vanilla_vlm/all_embed_impressions_small.pt')
    chexpert_cols=list(chex.columns[1:])
    print('chexpert columns: ', chexpert_cols)
    #chex[chexpert_cols] = np.vectorize(convert)(chex[chexpert_cols])
    imp_chex = chex
    print(imp_chex.head())
    imp_filtered = imp_chex[imp_chex['image_id'].isin(list(embed_dict_pred.keys()))].reset_index(drop=True)
    print(imp_filtered.head())
    imp_filtered.to_csv('imp_filtered'+'.csv')
    if prop!=100:
        samples = draw_samples(imp_filtered,num_samples)
        #print(samples)
        print(len(samples))
        sampled_data = imp_filtered[imp_filtered['image_id'].isin(samples)]
        sampled_data.to_csv('sampled_data_prop'+str(int(prop))+'.csv')
    else:
        sampled_data = imp_filtered#[imp_filtered['image_id'].isin(samples)]
    new_imp_filtered=sampled_data.iloc[:,1:]
    y_vals = np.array(new_imp_filtered) 
    #y_vals = np.tile(y_vals, (2, 1)) # doubles the number of samples because we concat preds and gts
    print('y vals shape: ', y_vals.shape)

    lps_filt = {key: embed_dict_pred[key] for key in sampled_data['image_id']}
    print('lps_filt: ', len(lps_filt))
    # lps_filt_gt = {key: embed_dict_gt[key] for key in sampled_data['image_id']}
    # print('lps_filt_gt: ', len(lps_filt_gt))
    #embed_dict_gt = {key: embed_dict_gt[key] for key in imp_filtered['image_id']}
    # seq_filt = {key: seq_dict_gt[key] for key in sampled_data['image_id']}
    # seq_filt_pred = {key: seq_dict_pred[key] for key in sampled_data['image_id']}
    #wt_filt_pred = {key: weight_dict_pred[key] for key in imp_filtered['image_id']}
    #wt_filt_gt = {key: weight_dict_gt[key] for key in imp_filtered['image_id']}

    # tensors_gt = list(lps_filt_gt.values())

    tensors_pred=list(lps_filt.values())
    # seq_gt_list=list(seq_filt.values())
    # seq_gt_list = list(map(lambda x: x-1 , seq_gt_list))
    # seq_lens_pred=list(seq_filt_pred.values())
    #weight_values = list(wt_filt_gt.values())+list(wt_filt_pred.values())
    #weight_values = list(wt_filt_gt.values())+list(wt_filt_pred.values())
    
    path = '/home/debodeep.banerjee/clevr/R2Gen/surrogate/'

    torch.save(tensors_pred, path+'tensors_preds_emb_full_50.pt')
    #torch.save(tensors_gt, path+'tensors_gt_emb_full_50.pt')
    # torch.save(seq_gt_list, path+'seq_lens_gt_full_50.pt')
    # torch.save(seq_lens_pred, path+'seq_lens_pred_full_50.pt')
    torch.save(y_vals, path+'surrogate_gt_labels_full_50.pt')

    # images 
    samp_ids=list(sampled_data['image_id'])
    filtered_data = [record for record in data['train'] if record['image_id'] in samp_ids]
    image_paths=[i['image_path'] for i in filtered_data]
    image_vecs=[processed_image(i) for i in image_paths]
    # vecs=process_images_in_batches(im_check, 2048)
    # image_features.extend(features)
    print('images number: ', len(image_vecs))
    torch.save(image_vecs, path+'image_vecs.pt')
    print('image vectors saved')
if __name__ == '__main__':
    main()