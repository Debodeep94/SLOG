import logging
import os
from abc import abstractmethod
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import time
import random
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

def count (tensor_data):
    print(torch.nn.functional.one_hot (tensor_data.long()))
    counts=torch.nn.functional.one_hot (tensor_data.long()).sum (dim = 0)#.sum(dim=0)
    counts=counts.sum(dim=0)
    return counts

def count_weights(tensor_data, val):
    # Get counts for each label
    counts = count(tensor_data)
    # Compute weights as the ratio for the specified label index
    weights = counts[val] / counts
    # Clamp weights to a maximum of 8
    weights = torch.clamp(weights, max=8)
    return weights


def first_surr_split(tensor_X, sequence_X, weight_X, y_vals, train_ratio=0.9):
    """
    tensor_X: combined list of predicted logprobs and ground truth logprobs
    sequence_X: combined list of sequence length of predicted sentences and gt sentences
    weight_X: combined list of weights for predicted sentences and gt sentences
    y_vals: groung truth labels for impression section
    """
    num_samples = len(tensor_X)
    print('num sample: ', num_samples)
    # Calculate the number of samples for each split
    num_train_samples = int(train_ratio * num_samples)
    #num_val_samples = int(val_ratio * num_samples)
    num_test_samples = num_samples - num_train_samples# - num_val_samples

    # Split the synthetic inputs into tensors
    tensors_train = tensor_X[:num_train_samples]
    tensors_test = tensor_X[num_train_samples:]# + num_val_samples:]
    # Stack tensors along dimension 0
    train_x= torch.stack(tensors_train, dim=0)
    print('train shape:',train_x.size())
    test_x= torch.stack(tensors_test, dim=0)
    seq_train = sequence_X[:num_train_samples]
    seq_test = sequence_X[num_train_samples:]
    wt_train = weight_X[:num_train_samples]
    
    tensors_train_y = y_vals[:num_train_samples]
    tensors_test_y = y_vals[num_train_samples:]
    train_y=torch.tensor(tensors_train_y)
    test_y=torch.tensor(tensors_test_y)

    # convert sequence and weights to tensor
    seq_train = torch.tensor(seq_train)
    wt_train = torch.tensor(wt_train)
    #seq_val = torch.tensor(seq_val)
    seq_test = torch.tensor(seq_test)
    weights = count_weights(train_y) #weights for CE loss (neg/pos)

    # Create DataLoader for training set, validation set, and test set
    train_dataset = TensorDataset(train_x, train_y, seq_train, wt_train)
    test_dataset = TensorDataset(test_x, test_y, seq_test)
    input_size, output_size= train_x.size(-1), train_y.size(1)
    print(test_y.size())
    print(seq_test.size())
    print(test_x.size())
    return weights, input_size, output_size, train_dataset, test_dataset

def prepare_data(tensor_X_gt,tensor_X_pred, sequence_X_gt,sequence_X_pred, y_vals):

    # As of now we are not using weights as the model is trained with gt data
    X_gt= torch.stack(tensor_X_gt, dim=0)
    X_pred= torch.stack(tensor_X_pred, dim=0)
    Y=torch.tensor(y_vals)
    # convert sequence and weights to tensor
    seq_X_gt = torch.tensor(sequence_X_gt)
    seq_X_pred = torch.tensor(sequence_X_pred)
    #weight_X = torch.tensor(weight_X)
    #ce_weights = count_weights(y_vals)
    return X_gt, X_pred, Y, seq_X_gt, seq_X_pred



def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def accuracy(scores, targets):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    scores = torch.argmax(scores, dim=1)
    return accuracy_score(scores, targets)


class Metrics(object):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    def __init__(self, ref_columns):
        self.y_true = []
        self.y_pred = []
        self.cond_names = ref_columns
        for i in range(14):
            self.y_true.append([])
            self.y_pred.append([])

    def update(self, y_pred, y_true):
        # print(y_true.size())
        # print(y_pred.size())
        y_pred = torch.argmax(y_pred, dim=2)
        # print(y_pred.size())

        for i in range(len(self.cond_names)):
            self.y_true[i].append(y_true[:, i])
            self.y_pred[i].append(y_pred[:, i])


    def calculate_metrics(self):
        metrics = {}

        # Compute metrics for each condition
        for i in range(len(self.cond_names)):
            y_true = torch.cat(self.y_true[i])
            y_pred = torch.cat(self.y_pred[i])

            metrics['Positive Precision ' + self.cond_names[i]] = list(precision_score(y_true, y_pred, labels=[1], average=None, zero_division=0))[0]
            metrics['Positive Recall ' + self.cond_names[i]] = list(recall_score(y_true, y_pred, labels=[1], average=None, zero_division=0))[0]
            metrics['Positive F1 ' + self.cond_names[i]] = list(f1_score(y_true, y_pred, labels=[1], average=None, zero_division=0))[0]

            metrics['Uncertain Precision ' + self.cond_names[i]] = \
            list(precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0))[0]
            metrics['Uncertain Recall ' + self.cond_names[i]] = \
            list(recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0))[0]
            metrics['Uncertain F1 ' + self.cond_names[i]] = list(f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0))[
                0]

            # metrics['Micro Precision '+self.cond_names[i]] = precision_score(y_true, y_pred, average='micro')
            # metrics['Micro Recall ' + self.cond_names[i]] = recall_score(y_true, y_pred, average='micro')
            # metrics['Micro F1 ' + self.cond_names[i]] = f1_score(y_true, y_pred, average='micro')

        # Compute global metrics
        #print(f"printing outputs.....:{self.y_pred}")
        #print(f"printing gts.....:{self.y_true}")
        master_y_true = torch.cat([inner for outer in self.y_true for inner in outer])
        master_y_pred = torch.cat([inner for outer in self.y_pred for inner in outer])

        metrics['Micro Positive Precision'] = list(precision_score(master_y_true, master_y_pred, labels=[1], average=None, zero_division=0))[0]
        metrics['Micro Positive Recall'] = list(recall_score(master_y_true, master_y_pred, labels=[1], average=None, zero_division=0))[0]
        metrics['Micro Positive F1'] = list(f1_score(master_y_true, master_y_pred, labels=[1], average=None, zero_division=0))[0]

        metrics['Micro Uncertain Precision'] = \
        list(precision_score(master_y_true, master_y_pred, labels=[2], average=None, zero_division=0))[0]
        metrics['Micro Uncertain Recall'] = \
        list(recall_score(master_y_true, master_y_pred, labels=[2], average=None, zero_division=0))[0]
        metrics['Micro Uncertain F1'] = list(f1_score(master_y_true, master_y_pred, labels=[2], average=None, zero_division=0))[0]
        print('metrics: ', metrics)
        return metrics
    

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def convert(val):
    if val==-1:
        return  2# This is just for matching the chexpert with ground truth
    elif val == 1:
        return val
    else:
        return 0

def convert_array(array_data):
    # Use numpy's where function for element-wise transformations
    array_data = np.where(array_data == -1, 2, array_data)  # Convert -1 to 2
    array_data = np.where(array_data == 1, 1, 0)            # Convert other values to 0, keep 1 as is
    return array_data

def draw_samples(data, num_samples):
        start_time = time.time()
        classes = data.columns[1:]
        label_set = np.array(data[classes])
        examples = list(data['study_id'])
        example_dict=dict(zip(examples, label_set))
        print(f'example dict: {example_dict}')
        symptoms = {k:[] for k in classes}
        symp_list=list(symptoms.keys())
        for i in examples:
            if 1 in example_dict[i]:
                if 2 in example_dict[i]:
                    print('there is unambigious label')
                location = np.where(example_dict[i]==1)[0]
                #print(location)
                for j in location:
                    symptoms[symp_list[j]].append(i)
            else:
                print('No positive mention')
                print(f'study id: {i}')
                print(f'label vector: {example_dict[i]}')
            # pick one sample from bucket 0
        # check for the least occurrance of label 
        # Sample from that bucket
        # Repeat till we meet the required number
        #print(f'loaded buckets: {symptoms}')
        # print the size of the buckets
        #sample = []
        count=0
        sample =[]
        bucket_counter=0
        class_to_remove = ""
        while count < num_samples:
            try:
                #print(f"Bucket counter : {bucket_counter}")
                #print(f"symp_list[bucket_counter]: {symp_list[bucket_counter]}")
                #print(f"symptoms[symp_list[bucket_counter]]: {symptoms[symp_list[bucket_counter]]}")
                draw = random.sample(symptoms[symp_list[bucket_counter]],1)
            
                print(f'sample picked: {draw[0]}')
                sample.append(draw[0])
                #print(symp_list)
                for j in symp_list:
                    #print(j)
                    print(f'before removing the sample from class {j}: {len(symptoms[j])}')
                    try:
                        symptoms[j].remove(draw[0])
                    except:
                        print(f'id not available in class {j}')
                    print(f'after removing the sample from class {j}: {len(symptoms[j])}')
                
                count+=1
            except:
                #print(f'Bucket {symp_list[bucket_counter]} is empty after count {count}')
                symp_list.remove(symp_list[bucket_counter])
                #print(f'updated list of classes: {symp_list}, lenght: {len(symp_list)}')
            bucket_counter+=1
            if bucket_counter==(len(symp_list)):
                bucket_counter=0
            print('=================================')
        end_time = time.time()
        print(f'total time taken for sampling process: {end_time-start_time} seconds')
        return sample
def processed_image(file_path):
    image = Image.open(file_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image)
    return image

# Define dataset class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_tensor = processed_image(file_path)
        return image_tensor

# Define function to extract features from a batch of images
def image_feat_extractor(images):
    model = models.resnet50(pretrained=True)
    modules = list(model.children())[:-2]
    model = torch.nn.Sequential(*modules)
    patch_feats = model(images)
    batch_size, feat_size, _, _ = patch_feats.shape
    patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
    return patch_feats

# Define function to process images in batches
def process_images_in_batches(file_paths, batch_size):
    dataset = ImageDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    for batch_images in tqdm(dataloader):
        batch_features = image_feat_extractor(batch_images)
        all_features.append(batch_features)
    return torch.cat(all_features, dim=0)


def compute_mlc_f1(gt, pred, label_set):
    print('gt: ', gt)
    print('pred: ', pred)
    res_mlc = {}
    score = 0
    for i, label in enumerate(label_set):
        res_mlc['F1_' + label] = round(f1_score(gt[:, i], pred[:, i], zero_division=0)*100,2)
        score += res_mlc['F1_' + label]
        #print(f'{i}:{}')
    res_mlc['AVG_F1'] = score / len(label_set)

    res_mlc['F1_MACRO'] = round(f1_score(gt, pred, average="macro", zero_division=0)*100,2)
    res_mlc['F1_MICRO'] = round(f1_score(gt, pred, average="micro", zero_division=0)*100,2)
    res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro", zero_division=0)
    res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro", zero_division=0)
    res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro", zero_division=0)
    res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro", zero_division=0)

    return res_mlc