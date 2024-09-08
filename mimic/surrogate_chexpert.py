import logging
import os
from abc import abstractmethod
import json
import pandas as pd
import ast
import cv2
import torch
#import #wandb
import numpy as np
from modules.utils import generate_heatmap, surrogate_regression, surrogate_split, convert
from surrogate import SurrogateModel, SurrogateLoss, CustomRidgeLoss, RidgeRegression
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import itertools
import random
#from torchviz import make_dot
#from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.utils import EarlyStopping
import csv
import torch 
torch.manual_seed(123)

# creating data for surrogate model
#def chex_surr(tensor_dict):
#print('hello')
chex=pd.read_csv('/home/debodeep.banerjee/chexpert-labeler/round_1_more_dara222.csv')
chex = chex.iloc[:,1:]
chex_cols=list(chex.columns)
chex[chex_cols] = np.vectorize(convert)(chex[chex_cols])
chex['info_score']=chex[chex_cols].sum(axis=1)
print(chex.head())
#print('hello')
with open('tensor_dict_r1_more_data.pt', 'rb') as file:
      tensor_dict = torch.load(file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#tensors= list(tensor_dict.values())
#print(tensors)
tensors= list(tensor_dict.values())
train_x=tensors[:int(len(tensors)*0.80)]
score_bucket = list(chex['info_score'])
train_y=np.array(score_bucket[:int(len(score_bucket)*0.80)])
#print('raw train y: ', train_y)
test_x=tensors[int(len(tensors)*0.80):]
test_y=np.array(score_bucket[int(len(score_bucket)*0.80):])
#print('train_x', train_x)
print('length of train x:',len(train_x))
print('length of train y:',len(train_y))


#minmaxscaler
scaler_trainX = MinMaxScaler()
scaler_trainY = MinMaxScaler()
scaler_test = MinMaxScaler()
# transform data

scaler_weight = MinMaxScaler()
# Redefine the train and validation
train_x = train_x[:int(len(train_x)*0.70)]
train_y = train_y[:int(len(train_y)*0.70)]

val_x = train_x[int(len(train_x)*0.70):]
val_y = train_y[int(len(train_y)*0.70):]

train_list=[]
for j in train_x:
    #i=i.cpu()
    new_list=j.tolist()
    train_list.append(new_list)
train_list=np.array(train_list)

val_list=[]
for j in val_x:
    #i=i.cpu()
    new_list=j.tolist()
    val_list.append(new_list)
val_list=np.array(val_list)

test_list=[]
for i in test_x:
    #i=i.cpu()
    new_list=i.tolist()
    test_list.append(new_list)
test_list=np.array(test_list)


trainX_to_reshape = np.array(train_list.reshape(len(train_list),512))
train_x = scaler_trainX.fit_transform(trainX_to_reshape)
#weight_train = scaler_weight.fit_transform(weight_train)
train_y = scaler_trainY.fit_transform(train_y.reshape(len(train_y),1))
#print('scaled train y: ', train_y)
# Extract the minimum and maximum values from the scaler
min_value = scaler_trainX.data_min_
max_value = scaler_trainX.data_max_

# Convert the values to tensors
min_tensor = torch.tensor(min_value)
max_tensor = torch.tensor(max_value)

# Save the tensors
torch.save({'min': min_tensor, 'max': max_tensor}, 'surr_chex_min_max_only_find_r1_more_data.pt') #train_min_max_scalar_seq_40: see att_model.py. We take the seq_40 

val_x = scaler_trainX.transform(val_list.reshape(len(val_list),512))
val_y = scaler_trainY.transform(val_y.reshape(len(val_y),1))

test_x = scaler_trainX.transform(test_list.reshape(len(test_list),512))
test_y = scaler_trainY.transform(test_y.reshape(len(test_y),1))
#print('scales_test_y: ', test_y)

#len(train_y_pt)
# Assume you have independent variables X and a dependent variable y

train_x_pt = torch.tensor(train_x,dtype=torch.float) #pt: pytorch
train_x_pt = train_x_pt/torch.norm(train_x_pt, dim = 1, keepdim= True)
train_y_pt = torch.tensor(train_y, dtype=torch.float).reshape(-1,1) #pt:pytorch
#print('train_x_pt: ', train_x_pt)

val_x_pt = torch.tensor(val_x,dtype=torch.float) #pt: pytorch
val_x_pt = val_x_pt/torch.norm(val_x_pt, dim = 1, keepdim= True)
val_y_pt = torch.tensor(val_y, dtype=torch.float).reshape(-1,1) #pt:pytorch

test_x_pt = torch.tensor(test_x,dtype=torch.float)
test_x_pt = test_x_pt/torch.norm(test_x_pt, dim = 1, keepdim= True)
test_y_pt = torch.tensor(test_y, dtype=torch.float).reshape(-1,1)
#print('train_y_pt: ', test_y_pt)
print('train_x', len(train_x))
print('val_x',len(val_x))
print('test_x',len(test_x))
weight_train = torch.ones(len(train_x))
print('train_x', len(train_x))
print('val_x',len(val_x))
print('test_x',len(test_x))


# Instantiate the model
input_dim = train_x_pt.size(1)
output_dim = train_y_pt.size(1)
model_surr = SurrogateModel(input_dim, 3)
# Define the loss function and optimizer
criterion1 = torch.nn.MSELoss()
criterion2 = SurrogateLoss()
optimizer = torch.optim.Adam(model_surr.parameters(), lr=0.0008)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, verbose=True)
early_stopping = EarlyStopping(patience=50, delta=0)

# Train the model
num_epochs = 300
train_loss = []
validation_loss = []
for epoch in range(num_epochs):
    #print('epochs: ', epoch)
    y_hat = model_surr(train_x_pt)
    #print('yhat: ', y_hat)
    loss = criterion2(weight_train, y_hat, train_y_pt)
    loss.backward()
    train_loss.append(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    
    y_hat_val = model_surr(val_x_pt)
    val_loss = criterion1(y_hat_val, val_y_pt)
    validation_loss.append(val_loss.item())
    #wandb.log({"train loss class reg": loss, "validation loss class reg": val_loss})
    if ((epoch+1)%10)==0:
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), val_loss.item()))

    scheduler.step(val_loss)
    early_stopping(val_loss)

    if early_stopping.early_stop:
        print("Early stopping")
        break
# Test the model
with torch.no_grad():
    predicted = model_surr(test_x_pt)
    #print('Predicted values: ', predicted)

#y_preds=np.asarray(scaler_test.inverse_transform(predicted))
#gt=torch.tensor(scaler_test.inverse_transform(test_y_pt))
#print(test_y)
#print('predicted', predicted)
mse= criterion1(torch.tensor(predicted), test_y_pt)
rel_rse=torch.sum((torch.tensor(predicted)- test_y_pt)**2)/torch.sum((test_y_pt-torch.mean(test_y_pt))**2)
rmse=mse**0.5
print('the rmse loss is ', rmse)
print('Saving surrogate model...')
torch.save(model_surr, 'surrogate/'+'surr_chex_lin_reg_split_only_find_r1_more_data.pt')
print('variance of test list: ', torch.std(test_y_pt))
print('variance of pred list: ', np.std(np.array(predicted)))
print('relative RSE: ', rel_rse)

# Plot the curves
plt.style.use('ggplot')
fig_surr=plt.figure(figsize=(16,9))
plt.rcParams.update({'font.size': 30})
plt.plot(train_loss, label= 'train loss', color='red', linewidth=2.0)
plt.plot(validation_loss, label= 'validation loss', color='blue', linewidth=2.0)
plt.legend()
#plt.title('Surrogate train on splitted training data')
plt.savefig('plots/surr_chex_lin_reg_split_only_find_r1_more_data.png')
plt.close(fig_surr)

############ Ridge regression ############

# Define the hyperparameters
input_size = train_x_pt.size(1)
alpha = 0.1
epochs = 300
lr = 0.002
ridge_loss=[]
ridge_val_loss = []
# Initialize the Ridge Regression model
model_ridge = RidgeRegression(input_size, alpha)

# Define the loss function
loss_fn = CustomRidgeLoss(alpha)

# Define the optimizer
optimizer = torch.optim.Adam(model_ridge.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
best_loss = float('inf')
counter = 0  # Counter to track the number of epochs without improvement
early_stopping = EarlyStopping(patience=50, delta=0)
#scheduler = StepLR(optimizer, step_size=100, gamma=0.8)  # Reduce the learning rate by a factor of 0.5 every 20 epochs

# Training loop
for epoch in range(epochs):
    # Forward pass
    outputs = model_ridge(train_x_pt)
    loss = loss_fn(model_ridge, weight_train,outputs.squeeze(), train_y_pt)  # Remove extra dimensions
    ridge_loss.append(loss.item())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #scheduler.step()
    with torch.no_grad():
        val_out=model_ridge(val_x_pt)
        val_loss=(criterion1(val_out.squeeze(), val_y_pt.reshape(-1)))
        ridge_val_loss.append(val_loss)
    #wandb.log({"train loss ridge reg": loss, "validation loss ridge reg": val_loss})
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, val_loss: {val_loss.item():.4f}')
    scheduler.step(val_loss)
    early_stopping(val_loss)

    if early_stopping.early_stop:
        print("Early stopping")
        break
# Retrieve the learned coefficients
coefficients = model_ridge.linear.weight.detach().numpy()
intercept = model_ridge.linear.bias.detach().numpy()
# Evaluate the model on the test set
with torch.no_grad():
    test_outputs = model_ridge(test_x_pt)
    test_loss = (criterion1(test_outputs.squeeze(), test_y_pt.reshape(-1)))**0.5
#print('ridge outputs: ', test_outputs)
print('the rmse loss is: ', test_loss)
rel_rse_ridge=torch.sum((torch.tensor(test_outputs)- test_y_pt)**2)/torch.sum((test_y_pt-torch.mean(test_y_pt))**2)
print("rel_rse_ridge: ", rel_rse_ridge)
torch.save(model_ridge, 'surrogate/'+'surr_chex_ridge_reg_split_only_find_r1_more_data.pt')

plt.style.use('ggplot')
fig_ridge=plt.figure(figsize=(16,9))
plt.rcParams.update({'font.size': 30})
plt.plot(ridge_loss, label='train loss', color='red', linewidth=2.0)
plt.plot(ridge_val_loss, label='validation loss', color='blue', linewidth=2.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend()
plt.savefig('plots/surr_chex_ridge_reg_split_only_find_r1_more_data.png')
plt.close(fig_ridge)
