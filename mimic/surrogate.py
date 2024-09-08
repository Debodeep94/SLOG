import torch
import pandas as pd 
import numpy as np
import torch.nn as nn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import itertools
import pickle
from torch import Tensor
import torch.nn._reduction as _Reduction
from typing import Callable, Optional
import torch.nn.functional as F



# Define the model
class SurrogateModel(torch.nn.Module):
    def __init__(self, input_dim,hidden_size):
        super(SurrogateModel, self).__init__()
        #self.linear = torch.nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        #out = nn.functional.relu(out)
        out = self.fc2(out)
        return out

class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class SurrogateLoss_(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, weights: Tensor, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)

class SurrogateLoss(nn.Module):
    def __init__(self):
        super(SurrogateLoss, self).__init__()

    def forward(self, predicted, target, weights):
        squared_error = torch.pow(predicted - target, 2)
        weighted_squared_error = squared_error * weights
        weighted_mse = torch.mean(weighted_squared_error)
        return weighted_mse

# Ridge regression

# Define the custom loss function
class CustomRidgeLoss(nn.Module):
    def __init__(self, alpha):
        super(CustomRidgeLoss, self).__init__()
        self.alpha = alpha

    def forward(self, model, weights,inputs, targets):
        loss = torch.mean(weights*(inputs - targets) ** 2)  # Custom loss calculation
        l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())
        loss += self.alpha * l2_reg
        return loss

# Define the Ridge Regression model
class RidgeRegression(nn.Module):
    def __init__(self, input_size, alpha):
        super(RidgeRegression, self).__init__()
        #self.embed = nn.Embedding.from_pretrained('target_embedding_weights.pth', freeze=True)
        self.linear = nn.Linear(input_size, 1)
        self.alpha = alpha

    def forward(self, x):
        #x=self.embed(captions)
        #x=torch.mean(x,dim=1)
        return self.linear(x)

    def predict(self, x):
        #x=self.embed(captions)
        #x=torch.mean(x,dim=1)
        return self.forward(x)