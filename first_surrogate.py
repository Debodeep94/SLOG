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




path = '/home/debodeep.banerjee/R2Gen/surrogate_vectors/with_lps/'
weight_box_gt=torch.load( path+'weight_box_gt.pt')
tensor_dict=torch.load( path+'tensor_dict.pt')
tensor_dict_gt=torch.load(path+'tensor_dict_gt.pt')
weight_dict_pred=torch.load( path+'weight_dict_pred.pt')
weight_dict_gt=torch.load( path+'weight_dict_gt.pt')
seq_dict_gt=torch.load( path+'seq_dict_gt.pt')
seq_dict_pred=torch.load(path+'seq_dict_pred.pt')
print(torch.cuda.memory_summary())
# Surrogate formation
chex = pd.read_csv('/home/debodeep.banerjee/chexpert-labeler/impression_chex.csv')
chex = chex.drop([ 'Reports'], axis=1) #'No Finding','Support Devices',
chex=chex.dropna(how='all')#.reset_index(drop=0)
indices= list(chex.index)


with open(r"/home/debodeep.banerjee/R2Gen/data/mimic/imp_n_find_split.json", 'r') as f:
    # Load the contents of the file into a variable
    data = json.load(f)

study_ids=[]
for i in data:
    for j in data[i]:
        study_ids.append(j['study_id'])
filtered_s_id=[study_ids[i] for i in indices]
chex.insert(0, 'study_id', filtered_s_id)

# train portion of the reports
study_ids_train=[]
for i in data['train']:
    #for j in data[i]:
    study_ids_train.append(i['study_id'])

# Further filter for the train data
chex = chex[chex['study_id'].isin(study_ids_train)]
#nan counts
nan_count = (chex.isna().sum()/chex.shape[0])*100
print(nan_count)
# remove columns with more than 95% nans
#chex = chex.drop(['Pleural Other','Fracture','Lung Lesion', 'Enlarged Cardiomediastinum'], axis=1)
chex=chex.fillna(0)
chex.head()

def binary(val):
    if val == -1:
        return 2
    
    else:
        return val
#embed_dict = torch.load('/home/debodeep.banerjee/R2Gen/all_embed_impressions_small.pt')
chexpert_cols=list(chex.columns[1:])

chex[chexpert_cols] = np.vectorize(binary)(chex[chexpert_cols])
imp_chex = chex
imp_filtered = imp_chex[imp_chex['study_id'].isin(list(tensor_dict.keys()))].reset_index(drop=True)
new_imp_filtered=imp_filtered.iloc[:,1:]
y_vals = np.array(new_imp_filtered) 
y_vals = np.tile(y_vals, (2, 1))
print('y vals shape: ', y_vals.shape)
lps_filt = {key: tensor_dict[key] for key in imp_filtered['study_id']}
print('lps_filt: ', len(lps_filt))
lps_filt_gt = {key: tensor_dict_gt[key] for key in imp_filtered['study_id']}
print('lps_filt_gt: ', len(lps_filt_gt))

# deal with embeddings
#embd_filt_gt = {key: embed_dict_gt[key] for key in imp_filtered['study_id']}
#embd_filt_pred = {key: embed_dict_pred[key] for key in imp_filtered['study_id']}
#print('embd-pred:', len(embd_filt_pred))
#print('embd-gt:', len(embd_filt_gt))
# Merging ground truth and preds
#merged_lps = {**lps_filt_gt, **lps_filt}
#print('lps-merged:', len(merged_lps))

seq_filt = {key: seq_dict_gt[key] for key in imp_filtered['study_id']}
seq_filt_pred = {key: seq_dict_pred[key] for key in imp_filtered['study_id']}
#gt_pred_merged_seq = {**seq_filt, **seq_filt_pred}
#seq_filt
wt_filt_pred = {key: weight_dict_pred[key] for key in imp_filtered['study_id']}
wt_filt_gt = {key: weight_dict_gt[key] for key in imp_filtered['study_id']}
#gt_pred_merged_wt = {**wt_filt_gt, **wt_filt_pred}
#samp_lps_filt = {key: tensor_dict_lps[key] for key in imp_filtered['study_id']}

tensors = list(lps_filt_gt.values())+list(lps_filt.values())
#print('tensors: ', tensors)
#tensors = torch.stack(tensors, dim=0)
seq_lens=list(seq_filt.values())+list(seq_filt_pred.values())
weight_values = list(wt_filt_gt.values())+list(wt_filt_pred.values())
print('tensor length: ', len(tensors))
# train test val split
# Split proportions
train_ratio, test_ratio = 0.9, 0.1
num_samples = len(tensors)
print('num sample: ', num_samples)
# Calculate the number of samples for each split
num_train_samples = int(train_ratio * num_samples)
#num_val_samples = int(val_ratio * num_samples)
num_test_samples = num_samples - num_train_samples# - num_val_samples

# Split the synthetic inputs into tensors
tensors_train = tensors[:num_train_samples]
# Find the maximum size along dimension 0
#max_size = max(t.size(0) for t in tensors_train)

# Pad or trim each tensor to match the maximum size
#tensors_padded = [F.pad(t, pad=(0, 0, 0, max_size - t.size(0)), value=0) for t in tensors_train]

# Stack tensors along dimension 0
train_x= torch.stack(tensors_train, dim=0)
print('train shape:-',train_x.size())
#tensors_val = tensors[num_train_samples:num_train_samples + num_val_samples]
#val_x= torch.stack(tensors_val, dim=0)
tensors_test = tensors[num_train_samples:]# + num_val_samples:]
#print('tensors test: ', tensors_test)
#max_size = max(t.size(0) for t in tensors_test)

# Pad or trim each tensor to match the maximum size
#tensors_padded = [F.pad(t, pad=(0, 0, 0, max_size - t.size(0)), value=0) for t in tensors_test]

test_x= torch.stack(tensors_test, dim=0)

tensors_train_y = y_vals[:num_train_samples]
#tensors_val_y = y_vals[num_train_samples:num_train_samples + num_val_samples]
tensors_test_y = y_vals[num_train_samples:]

seq_train = seq_lens[:num_train_samples]
#seq_val = seq_lens[num_train_samples:num_train_samples + num_val_samples]
seq_test = seq_lens[num_train_samples:]

wt_train = weight_values[:num_train_samples]

train_y=torch.tensor(tensors_train_y)
#val_y=torch.tensor(tensors_val_y)
test_y=torch.tensor(tensors_test_y)

seq_train = torch.tensor(seq_train)
wt_train = torch.tensor(wt_train)
#seq_val = torch.tensor(seq_val)
seq_test = torch.tensor(seq_test)
weights = count_weights(train_y)
#weights_val = count_weights(val_y)
# Model training
torch.manual_seed(568)
print('printing tensor sizes')
print(train_x.size())
print(train_y.size())
#print(val_x.size())
print(test_x.size())
input_size = train_x.size(-1)  # Vocabulary size
hidden_size = 1024   # Number of LSTM units
num_layers = 8     # Number of LSTM layers
output_size = train_y.size(1)     # Number of output classes
batch_size = 512     # Batch size for training, validation, and testing
learning_rate = 2e-4
num_classes = train_y.size(1)
num_labels = 3
num_heads = 14
#metrics = Metrics()

metrics = Metrics(chexpert_cols)
print_freq = 5

# Create DataLoader for training set, validation set, and test set
train_main_dataset = TensorDataset(train_x, train_y, seq_train, wt_train)
#train_loader = DataLoader(train_main_dataset, batch_size=batch_size, shuffle=True)

#val_dataset = TensorDataset(val_x, val_y, seq_val)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(test_y.size())
print(seq_test.size())
print(test_x.size())
test_dataset = TensorDataset(test_x, test_y, seq_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#print('train_x_su1: ', train_x)
# Instantiate the model and move it to GPU if available

dataset_size = len(train_main_dataset)
num_folds = 5
fold_size = dataset_size // num_folds
val_accs=[]
sur2_data=[]
val_seq = []
all_val_loss = []
all_val_f1 = []
all_train_loss = []
torch.cuda.empty_cache()
for fold in range(num_folds):
    print(f'entering fold: {fold}. The dataset size is {dataset_size}')
    start = fold * fold_size
    end = start + fold_size
    train_indices = list(set(range(dataset_size)))

    train_dataset = torch.utils.data.Subset(train_main_dataset, train_indices)

    train_size = int(0.8 * len(train_dataset))
    validation_size = len(train_dataset) - train_size
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
    print('val dataset:', validation_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM_Attn(input_size, hidden_size,num_labels)#, num_classes, num_heads)
    model=model.to(device)

    # Define loss function and optimizer

    # Instantiate the custom loss function
    criterion = torch.nn.CrossEntropyLoss(weight=(weights).to(device))#, reduction='none')
    criterion2 = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Number of training epochs
    num_epochs = 500
    val_loss_box=[]
    train_loss_box=[]
    val_f1 = []
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    current_patience = 0
    best_f1 = 0.0
    best_recall = 0.0
    best_prec = 0.0
    # Training and validation loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_samples = 0

        for batch_inputs, batch_targets, batch_seq_lens, batch_wt_train in train_loader:
            batch_inputs, batch_targets, batch_seq_lens, batch_wt_train = batch_inputs.to(device), batch_targets.to(device), batch_seq_lens.to(device), batch_wt_train.to(device)
            
            # Modify target tensor to have the correct shape
            #batch_targets_modified = torch.zeros(batch_targets.size(0), batch_targets.size(1), 2).to(device)
            #batch_targets_modified.scatter_(2, batch_targets.unsqueeze(-1), 1)
            #print('batch_targets_modified:', batch_targets_modified)
            outputs = model(batch_inputs, batch_seq_lens)
            #specific_output=train_y[:,3]
            # Apply sigmoid activation and reshape the outputs
            #sigmoid_outputs = torch.softmax(outputs, dim=1)#.view(batch_targets.size(0), -1, 2)
            temp1 = outputs.view(-1, num_labels)
            #print(temp1.size())
            batch_targets=batch_targets.to(torch.int64)
            temp2 = batch_targets.view(-1)#.to(torch.float32)
            #print(temp1.size(), temp2.size())
            loss = torch.mean(batch_wt_train)*criterion(temp1, temp2)
            
            #print(loss)
            total_loss += loss.item() * batch_inputs.size(0)
            
            total_samples += batch_targets.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clip_gradient(optimizer, 0.8)
        average_loss = total_loss / total_samples
        train_loss_box.append(average_loss)
        # Print training loss for the current epoch
        

        # Validation phase
        model.eval()
        total_correct = 0
        total_samples = 0

        all_acc=[]
        val_inputs=[]
        val_batch_seq=[]
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
        metrics = Metrics(chexpert_cols)

        print('entering validation')
        start = time.time()
        with torch.no_grad():
            for (batch_inputs, batch_targets, batch_seq_lens, batch_wt_train) in val_loader:
                # store them before moving to the device
                val_inputs.append(batch_inputs.to('cpu'))
                val_batch_seq.append(batch_seq_lens.to('cpu'))

                batch_inputs, batch_targets, batch_seq_lens = batch_inputs.to(device), batch_targets.to(device), batch_seq_lens.to(device)
                outputs = model(batch_inputs, batch_seq_lens)
                #print('output: ', outputs.size())
                individual_outputs = torch.argmax(outputs, dim=2) # convert to nominals
                #sigmoid_matrices.append(outputs)
                #full_batch.append(batch_targets)
                
                val_loss = criterion2(outputs.view(-1, num_labels), 
                                    batch_targets.to(torch.int64).view(-1))
                
                total_val_loss += val_loss.item() * batch_inputs.size(0)
                total_predictions = outputs.size(0)*outputs.size(1)
                losses.update(val_loss.item(), total_predictions)
                total_val_samples += batch_targets.size(0)
                correct_predictions = (batch_targets == individual_outputs).float() # .sum(dim=1).float() is want sum
                all_acc.append(correct_predictions.to('cpu'))
                acc = accuracy(outputs.view(-1, 3).to('cpu'), batch_targets.view(-1).to('cpu'))
                print('batch wise accuracy: ', acc)
                accs.update(acc, total_predictions)
                batch_time.update(time.time() - start)
                metrics.update(outputs.to('cpu'), batch_targets.to('cpu'))
                
                # Print status
                count_iter+=1
                start = time.time()

                # Print status
                if count_iter % print_freq == 0:
                    print('Validation: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top-5 Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                    loss=losses, acc=accs))
        
        val_loss_box.append(total_val_loss/total_val_samples)
        metrics_dict = metrics.calculate_metrics()
        print(
            '\n * LOSS - {loss.avg:.3f}\n'.format(
                loss=losses))
        pos_f1 = metrics_dict['Micro Positive F1']
        val_f1.append(pos_f1)
        print('positive f1:',  pos_f1)
        if pos_f1 > best_f1:# and recall > best_recall:
            best_f1 = pos_f1
            
            #best_recall = recall
            # Save the model
            torch.save(model.state_dict(), f'best_model_surr1_tial_lps_no_gumbel_{fold}.pth')
            print('Model saved! best f1: {:.4f}'.format(best_f1))
            current_patience = 0  # Reset patience counter
        else:
            current_patience += 1
            print('###################################################')
        if current_patience >= patience:
            print(f'Validation recall has not improved for {patience} epochs. Stopping training.')
            val_inputs_ = torch.cat(val_inputs, dim = 0)
            print('val_inputs shape:', val_inputs_.size())
            val_seq_ = torch.cat(val_batch_seq, dim = 0)
            all_acc_ = torch.cat(all_acc, dim =0)
            val_accs.append(all_acc_)
            sur2_data.append(val_inputs_)
            val_seq.append(val_seq_)
            print(f'model saved. size of surr 2 training data {len(sur2_data)}')
            print(f'printing surr 2 training data: {sur2_data}')
            # saving data for surrogate 2
            torch.save(sur2_data, path+'sur2_data.pt')
            torch.save(val_accs, path+'val_accs.pt')
            torch.save(val_seq, path+'val_seq.pt')
            val_inputs.clear()
            all_acc.clear()
            val_batch_seq.clear()

            break
    # storing all validation and training metrics
    all_val_loss.append(val_loss_box)
    all_train_loss.append(train_loss_box)
    all_val_f1.append(val_f1)
# Testing phase
for fold in range(num_folds):
    criterion = torch.nn.CrossEntropyLoss()
    # Instantiate the model and move it to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = LSTM_Attn(input_size, hidden_size,num_labels)#,num_layers, output_size,3,dropout_rate=0.2)
    loaded_model = loaded_model.to(device)
    checkpoint = torch.load(f'best_model_surr1_tial_lps_no_gumbel_{fold}.pth')
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
    total_val_loss = 0
    total_val_samples = 0
    count_iter=0
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    accs = AverageMeter()  # accuracy
    metrics = Metrics(chexpert_cols)

    start = time.time()
    with torch.no_grad():
        for (batch_inputs, batch_targets, batch_seq_lens) in test_loader:
            batch_inputs, batch_targets, batch_seq_lens = batch_inputs.to(device), batch_targets.to(device), batch_seq_lens.to(device)
            outputs = loaded_model(batch_inputs, batch_seq_lens)
            individual_outputs = torch.argmax(outputs, dim=2)
            #sigmoid_outputs = torch.softmax(outputs, dim=1)
            #sigmoid_matrices.append(outputs)
            full_batch.append(batch_targets)
            #print(outputs)
            #print('ground truth: ', batch_targets)
            #print('sigmoid outputs: ', sigmoid_outputs)
            val_loss = criterion2(outputs.view(-1, num_labels), 
                                    batch_targets.to(torch.int64).view(-1))
            
            total_val_loss += val_loss.item() * batch_inputs.size(0)
            total_predictions = outputs.size(0)*outputs.size(1)
            losses.update(val_loss.item(), total_predictions)
            total_val_samples += batch_targets.size(0)
            # Calculate sample-wise accuracy
            correct_predictions = (batch_targets == individual_outputs).float() # .sum(dim=1).float() is want sum
            total_predictions = individual_outputs.shape[1]
            #judge = [1 if batch_targets[i] == individual_outputs[i] else 0 for i in range(len(batch_targets))]
            #sample_accuracy = correct_predictions / total_predictions
            #print('sample wise accuracy: ', sample_accuracy)
            all_acc.append(correct_predictions)
            acc = accuracy(outputs.view(-1, 3).to('cpu'), batch_targets.view(-1).to('cpu'))
            
            #print('batch wise accuracy: ', accs)
            accs.update(acc, total_predictions)
            batch_time.update(time.time() - start)
            metrics.update(outputs.to('cpu'), batch_targets.to('cpu'))
            # Print status
            count_iter+=1
            start = time.time()

            # Print status
            if count_iter % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top-5 Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, acc=accs))
    flat_acc_list = torch.stack([item for sublist in all_acc for item in sublist])
    #print('flat_acc_list: ',flat_acc_list)
    metrics_dict = metrics.calculate_metrics()
    print(
        '\n * LOSS - {loss.avg:.3f}\n'.format(
            loss=losses))
    pos_f1 = metrics_dict['Micro Positive F1']
    print('all metrics: ', metrics_dict)
    print ( 'post_f1: ', pos_f1)
# Plot the necessary curved

# Surrogate 1
#plt.style.use('ggplot')
fig_sur1_loss=plt.figure(figsize=(16,9))
plt.rcParams.update({'font.size': 30})
# Pad lists with None for different lengths
pad_value = None
fold_train_losses_padded = [seq + [pad_value] * (max(map(len, all_train_loss)) - len(seq)) for seq in all_train_loss]
fold_val_losses_padded = [seq + [pad_value] * (max(map(len, all_val_loss)) - len(seq)) for seq in all_val_loss]

# Plot the training and validation losses with different tones of red and blue
for i, (train_loss, val_loss) in enumerate(zip(fold_train_losses_padded, fold_val_losses_padded)):
    epochs = np.arange(1, len(train_loss) + 1)

    # Plot training loss in different tones of red
    color_train = plt.cm.Reds(0.2 + 0.2 * i)  # Adjust 0.2 and 0.2 for different tones
    plt.plot( train_loss, label=f'Fold {i + 1} - Train', color=color_train, alpha=0.7)

    # Plot validation loss in different tones of blue
    color_val = plt.cm.Blues(0.2 + 0.2 * i)  # Adjust 0.2 and 0.2 for different tones
    plt.plot( val_loss, label=f'Fold {i + 1} - Validation', color=color_val, alpha=0.7)

plt.xlabel('x-axis')
plt.ylabel('Loss')
plt.title('Training and Validation Losses under 5-fold Cross-Validation')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('loss curve for surrogate 1')
plt.savefig('plots/surrogate_1_loss_5_fold.png')
plt.close(fig_sur1_loss)'''