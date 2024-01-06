import logging
import os
from abc import abstractmethod
import json
import pandas as pd
import ast
import cv2
import torch
import wandb
import numpy as np
from modules.utils import generate_heatmap, convert#, surrogate_regression, surrogate_split
from surrogate import SurrogateModel, SurrogateLoss, CustomRidgeLoss, RidgeRegression
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import itertools
import random
from modules.rnn_surrogate import *#MultiLabelLSTMModel, IndependentMultiLabelLSTMModel, AttentionLSTMClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.utils import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
import csv
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import time
from modules.surrogate_utils import *
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy


# code for second surrogate
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
path = '/home/debodeep.banerjee/R2Gen/surrogate_vectors/with_lps/'
sur2_data=torch.load( path+'sur2_data.pt')
val_accs=torch.load( path+'val_accs.pt')
val_seq=torch.load(path+'val_seq.pt')
surr_2_x = torch.stack([item for sublist in sur2_data for item in sublist])# torch.cat(sur2_data[0], dim=0)#test_x#torch.mean(test_x, dim=1)
print(torch.cuda.memory_summary())
print(f'sur2x: {surr_2_x.size()}')
surr_2_y = torch.stack([item for sublist in val_accs for item in sublist])
surr_2_seq = torch.stack([item for sublist in val_seq for item in sublist])

X = surr_2_x.cpu().detach().numpy()
Y = surr_2_y.cpu().detach().numpy()
seq_data = surr_2_seq.cpu().detach().numpy()



train_x=X[:int(len(X)*0.85)]
seq_tr=seq_data[:int(len(seq_data)*0.85)]
        
train_y=Y[:int(len(Y)*0.85)]

test_x_=X[int(len(X)*0.85):]
seq_test_=seq_data[int(len(seq_data)*0.85):] 
test_y_=Y[int(len(Y)*0.85):]

train_x_ = train_x[:int(len(train_x)*0.79)]
train_y_ = train_y[:int(len(train_y)*0.79)]
seq_tr_ = seq_tr[:int(len(seq_tr)*0.79)]
print(train_x_.shape)
#train_x = torch.nn.functional.normalize(train_x)
#train_y = torch.nn.functional.normalize(train_y)
#weight_train = torch.nn.functional.normalize(weight_train)

val_x_ = train_x[int(len(train_x)*0.79):]
val_y_ = train_y[int(len(train_y)*0.79):]
seq_val_= seq_tr[int(len(seq_tr)*0.79):]

train_x_pt = torch.tensor(train_x_,dtype=torch.float) #pt: pytorch
#train_x_pt = train_x_pt/torch.norm(train_x_pt, dim = 1, keepdim= True)
#print('train_x_surr2: ', train_x_pt)
seq_tr_=torch.tensor(seq_tr_, dtype=torch.int)
val_x_pt = torch.tensor(val_x_,dtype=torch.float) #pt: pytorch
#val_x_pt = val_x_pt/torch.norm(val_x_pt, dim = 1, keepdim= True)
seq_val_=torch.tensor(seq_val_, dtype=torch.int)
test_x_pt = torch.tensor(test_x_,dtype=torch.float)
#test_x_pt = test_x_pt/torch.norm(test_x_pt, dim = 1, keepdim= True)
seq_test_=torch.tensor(seq_test_, dtype=torch.int)
train_y_ = torch.tensor(train_y_)

val_y_ = torch.tensor(val_y_)
test_y_ = torch.tensor(test_y_)

weight_train = torch.ones(len(train_x_))

input_size = train_x_pt.size(-1)  # Vocabulary size
hidden_size = 1024   # Number of LSTM units
num_layers = 8     # Number of LSTM layers
output_size = train_y_.size(1)     # Number of output classes
#print('output_size: ',output_size)
batch_size = 512     # Batch size for training, validation, and testing
learning_rate = 2e-5
num_classes = train_y_.size(1)
num_labels = 2
num_heads = train_y_.size(1)
weights = count_weights(train_y_)
# Instantiate the model and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_2 = LSTM_Attn(input_size, hidden_size, num_labels)#, num_classes, num_heads)
model_2=model_2.to(device)

# Create DataLoader for training set, validation set, and test set
train_dataset = TensorDataset(train_x_pt, train_y_, seq_tr_)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_x_pt, val_y_, seq_val_)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(test_x_pt, test_y_, seq_test_)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Accuracy surrogate
# Instantiate the custom loss function
criterion = torch.nn.CrossEntropyLoss(weight=(weights).to(device))#, reduction='none')
criterion2 = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model_2.parameters(), lr=learning_rate)

# Number of training epochs
num_epochs = 500
val_loss_box_sur2=[]
train_loss_box_sur2=[]
f1_sur2=[]
patience = 25  # Number of epochs to wait for improvement
best_val_loss = float('inf')
current_patience = 0
best_f1 = 0.0
best_recall = 0.0
best_prec = 0.0
# Training and validation loop
for epoch in range(num_epochs):
    # Training phase
    model_2.train()
    total_loss = 0
    total_samples = 0

    for batch_inputs, batch_targets, batch_seq_lens in tqdm(train_loader):
        batch_inputs, batch_targets, batch_seq_lens = batch_inputs.to(device), batch_targets.to(device), batch_seq_lens.to(device)#, batch_wt_train.to(device)
        outputs = model_2(batch_inputs, batch_seq_lens)
        
        temp1 = outputs.view(-1, num_labels)
        #print('temp1: ', temp1)
        batch_targets=batch_targets.to(torch.float32)
        temp2 = batch_targets.view(-1).to(torch.int64)
        #print(temp1.size(), temp2.size())
        loss = criterion(temp1, temp2)
        
        #print(loss)
        total_loss += loss.item() #* batch_inputs.size(0)
        
        total_samples += batch_targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clip_gradient(optimizer, 1)
    average_loss = total_loss / len(train_loader)
    train_loss_box_sur2.append(average_loss)
    print('train loss: ',total_loss/(len(train_loader)))
    # Print training loss for the current epoch
    
    print('entering validation...')
    # Validation phase
    model_2.eval()
    total_correct = 0
    total_samples = 0

    all_acc=[]
    all_f1=[]
    all_rec=[]
    all_prec=[]
    sigmoid_matrices = []
    full_batch = []
    total_val_loss = 0
    total_val_samples = 0
    count_iter=0
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    accs = AverageMeter()  # accuracy
    pred_mat=[]
    start = time.time()
    with torch.no_grad():
        for (batch_inputs, batch_targets, batch_seq_lens) in tqdm(val_loader):
            batch_inputs, batch_targets, batch_seq_lens = batch_inputs.to(device), batch_targets.to(device), batch_seq_lens.to(device)
            outputs = model_2(batch_inputs, batch_seq_lens)
            #print('output: ', outputs)
            individual_outputs = torch.argmax(outputs, dim=2) # convert to nominals
            sigmoid_matrices.append(outputs)
            full_batch.append(batch_targets)
            pred_mat.append(individual_outputs)
            #print(outputs)
            #print('ground truth: ', batch_targets)
            #print('sigmoid outputs: ', sigmoid_outputs)
            val_loss = criterion2(outputs.view(-1, num_labels), 
                                batch_targets.to(torch.int64).view(-1))
            
            total_val_loss += val_loss.item() #* batch_inputs.size(0)
    final_val_loss = total_val_loss/ len(val_loader)
    val_loss_box_sur2.append(final_val_loss)
    print('final_val_loss: ', final_val_loss)

    # When the batch job is done, check for the metrics
    # As we have two labels, things are easier now. 
    gt = torch.stack([item for sublist in full_batch for item in sublist])
    all_preds = torch.stack([item for sublist in pred_mat for item in sublist])
    #print(all_preds)
    f_micro = f1_score(gt.cpu(), all_preds.cpu(), average="micro")
    f1_sur2.append(f_micro)
    if f_micro > best_f1:# and recall > best_recall:
        best_f1 = f_micro
        #best_recall = recall
        # Save the model
        torch.save(model_2.state_dict(), 'best_model_surr2_tial_lps_no_gumbel.pth')
        print('Model saved! best f1: {:.4f}'.format(best_f1))
        current_patience = 0  # Reset patience counter
    else:
        current_patience += 1
        print('###################################################')
    if current_patience >= patience:
        print(f'Validation f1 ({f_micro}) has not improved for {patience} epochs. Stopping training.')
        break

# Testing phase

criterion = torch.nn.CrossEntropyLoss()
# Instantiate the model and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model = LSTM_Attn(input_size, hidden_size, num_labels)#,num_layers, output_size,3,dropout_rate=0.2)
loaded_model = loaded_model.to(device)
checkpoint = torch.load('best_model_surr2_tial_lps_no_gumbel.pth')
loaded_model.load_state_dict(checkpoint)

loaded_model.eval()
total_correct = 0
total_samples = 0

all_acc=[]
all_f1=[]
all_rec=[]
all_prec=[]
sigmoid_matrices = []
full_batch = []
final_test_loss = 0
total_val_samples = 0
count_iter=0
pred_test = []
total_test_loss=0

start = time.time()
with torch.no_grad():
    for (batch_inputs, batch_targets, batch_seq_lens) in test_loader:
        batch_inputs, batch_targets, batch_seq_lens = batch_inputs.to(device), batch_targets.to(device), batch_seq_lens.to(device)
        outputs = loaded_model(batch_inputs, batch_seq_lens)
        
        individual_outputs = torch.argmax(outputs, dim=2)
        #sigmoid_outputs = torch.softmax(outputs, dim=1)
        sigmoid_matrices.append(outputs)
        full_batch.append(batch_targets)
        pred_test.append(individual_outputs)
        #print(outputs)
        #print('ground truth: ', batch_targets)
        #print('sigmoid outputs: ', sigmoid_outputs)
        test_loss = criterion2(outputs.view(-1, num_labels), 
                                batch_targets.to(torch.int64).view(-1))
        
        total_test_loss += test_loss.item()# * batch_inputs.size(0)
        #total_predictions = outputs.size(0)*outputs.size(1)
    final_test_loss = total_test_loss/ len(test_loader)    
    gt = torch.stack([item for sublist in full_batch for item in sublist])
    all_preds = torch.stack([item for sublist in pred_test for item in sublist])
    f_micro = f1_score(gt.cpu(), all_preds.cpu(), average="micro") 
    print('f_micro test: ', f_micro)  
    print('test loss: ', final_test_loss) 

# Plot the necessary curved


# Surrogate 2
plt.style.use('ggplot')
fig_sur2_loss=plt.figure(figsize=(16,9))
plt.rcParams.update({'font.size': 30})
plt.plot(train_loss_box_sur2, label='train loss', color='red', linewidth=2.0)
plt.plot(val_loss_box_sur2, label='validation loss', color='blue', linewidth=2.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend()
plt.title('loss curve for surrogate 2')
plt.savefig('plots/surrogate_2_loss.png')
plt.close(fig_sur2_loss)

# F1 @ surrogate2
plt.style.use('ggplot')
fig_sur2_f1=plt.figure(figsize=(16,9))
plt.rcParams.update({'font.size': 30})
plt.plot(f1_sur2, color='red', linewidth=2.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.legend()
plt.title('F1 curve for surrogate 2')
plt.savefig('plots/surrogate_2_f1_lps_val.png')
plt.close(fig_sur2_f1)