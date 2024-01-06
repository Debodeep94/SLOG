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



def count (tensor_data):
    counts=torch.nn.functional.one_hot (tensor_data.long()).sum (dim = 0)#.sum(dim=0)
    counts=counts.sum(dim=0)
    return counts

def count_weights(tensor_data):
    counts=count ( tensor_data)
    #print(counts)
    weights=counts[0]/counts
    weights = torch.clamp(weights, max=15)
    return weights#/300

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