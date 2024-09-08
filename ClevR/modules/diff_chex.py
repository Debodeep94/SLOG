import logging
import os
from abc import abstractmethod
import json
import pandas as pd
import ast
import cv2
import torch
import torch.nn as nn
import numpy as np
from modules.utils import generate_heatmap
from surrogate import SurrogateModel
from surrogate import SurrogateLoss
#from pycocoevalcap.bleu.bleu import Bleu
from modules.surrogate_utils import *
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch
import itertools
import csv
import torch
#from torchviz import make_dot
#from torchsummary import summary
import matplotlib.pyplot as plt

from modules.utils import convert#, surrogate_regression, surrogate_split
from surrogate import SurrogateModel, SurrogateLoss, CustomRidgeLoss, RidgeRegression
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import itertools
import random
from modules.rnn_surrogate import *#MultiLabelLSTMModel, IndependentMultiLabelLSTMModel, AttentionLSTMClassifier
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
class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        #print('printing the model')
        #print(self.model.visual_extractor)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        self.ann_path = args.ann_path
        self._load_checkpoint(args.load)
        self.info_score_data= args.info_score_data
    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        #surrogate_checkpoint= torch.load('/nfs/data_chaos/dbanerjee/my_data/R2Gen/surrogate/surr_model_sample_size_2500.pt')
        self.model.load_state_dict(checkpoint['state_dict'])
        #print(self.model)#
        #print(self.model.encoder_decoder.model.tgt_embed[0].lut.weight.data)
        # Save the embedding weights to a file
        target_embedding_weights= self.model.encoder_decoder.model.tgt_embed#[0].lut.weight.data
        print(target_embedding_weights)
        #torch.save(target_embedding_weights, 'target_embedding_weights.pth')

class SurrogateFamily(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, train_dataloader):
        super(SurrogateFamily, self).__init__(model, criterion, metric_ftns, args)
       # self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        #self.surrogate_dataloader = surrogate_dataloader
    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        img_ids, embed_box, seq_box, idx_box = [], [], [], []
        count=0
        for batch_idx, (images_id, images, reports_ids, reports_masks,seq_length) in tqdm(enumerate(self.train_dataloader)):
            images, reports_ids, reports_masks, seq_length = images.to(self.device), reports_ids.to(
                self.device), reports_masks.to(self.device), seq_length.to(self.device)
            
            # Load the saved embedding weights directly into the new embedding layer
            #saved_embedding_weights = torch.load('target_embedding_weights.pth')
            #new_embedding_layer=nn.Embedding.from_pretrained(saved_embedding_weights)
            #new_embedding_layer.weight.data.copy_(saved_embedding_weights)
            #new_embedding_layer=new_embedding_layer.to(self.device)
            target_embedding_weights= self.model.encoder_decoder.model.tgt_embed
            embedded_sentence = target_embedding_weights(reports_ids[:, 1:])
            embed_box.extend(embedded_sentence)
            idx_box.extend(reports_ids[:, 1:])
            seq_box.extend(seq_length)
            img_ids.extend(list(images_id))
            #count+=1
            #if count == 2:
                #break
        embed_dict = dict(zip(img_ids,embed_box))
        seq_dict = dict(zip(img_ids,seq_box))
        idx_dict=dict(zip(img_ids,idx_box))
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
        imp_filtered = imp_chex[imp_chex['study_id'].isin(list(embed_dict.keys()))].reset_index(drop=True)
        new_imp_filtered=imp_filtered.iloc[:,1:]
        y_vals = np.array(new_imp_filtered) 

        # filter the dictionary with keys available in the dataset
        embd_filt = {key: embed_dict[key] for key in imp_filtered['study_id']}
        seq_filt = {key: seq_dict[key] for key in imp_filtered['study_id']} #seq lengths
        idx_filt = {key: idx_dict[key] for key in imp_filtered['study_id']}
        tensors= list(embd_filt.values()) #tensor_dict: gumbel_softmax logps ; embed_dict: embedded outputs
        seq_lens=list(seq_filt.values())
        print('tensor: ', len(tensors))
        #print('tnsors: ', tensors)
        # train test val split
        # Split proportions
        train_ratio, val_ratio, test_ratio = 0.85, 0.05, 0.1
        num_samples = len(tensors)
        # Calculate the number of samples for each split
        num_train_samples = int(train_ratio * num_samples)
        num_val_samples = int(val_ratio * num_samples)
        num_test_samples = num_samples - num_train_samples - num_val_samples

        # Split the synthetic inputs into tensors
        tensors_train = tensors[:num_train_samples]
        tensors_val = tensors[num_train_samples:num_train_samples + num_val_samples]
        tensors_test = tensors[num_train_samples + num_val_samples:]

        tensors_train_y = y_vals[:num_train_samples]
        tensors_val_y = y_vals[num_train_samples:num_train_samples + num_val_samples]
        tensors_test_y = y_vals[num_train_samples + num_val_samples:]

        seq_train = seq_lens[:num_train_samples]
        seq_val = seq_lens[num_train_samples:num_train_samples + num_val_samples]
        seq_test = seq_lens[num_train_samples + num_val_samples:]
        #print('tensor train:', tensors_train)
        # Print the shapes of the split datasets as examples
        print("Train Shape:")
        print(len(tensors_train))
        print("Validation Shape:")
        print(len(tensors_val))
        print("Test Shape:")
        print(len(tensors_test))
        train_list=[]
        for j in tqdm(tensors_train):
            #i=i.cpu()
            new_list=j.tolist()
            train_list.append(new_list)
        train_list=np.array(train_list)
        train_x= torch.tensor(train_list)


        val_list=[]
        for j in tqdm(tensors_val):
            #i=i.cpu()
            new_list=j.tolist()
            val_list.append(new_list)
        val_list=np.array(val_list)
        val_x= torch.tensor(val_list)

        test_list=[]
        for j in tqdm(tensors_test):
            #i=i.cpu()
            new_list=j.tolist()
            test_list.append(new_list)
        test_list=np.array(test_list)
        test_x= torch.tensor(test_list)

        train_y=torch.tensor(tensors_train_y)
        val_y=torch.tensor(tensors_val_y)
        test_y=torch.tensor(tensors_test_y)

        seq_train = torch.tensor(seq_train)
        seq_val = torch.tensor(seq_val)
        seq_test = torch.tensor(seq_test)
        weights = count_weights(train_y)
        weights_val = count_weights(val_y)
        # Model training
        torch.manual_seed(568)

        input_size = train_x.size(-1)  # Vocabulary size
        hidden_size = 512   # Number of LSTM units
        num_layers = 8     # Number of LSTM layers
        output_size = train_y.size(1)     # Number of output classes
        batch_size = 256     # Batch size for training, validation, and testing
        learning_rate = 2e-4
        num_classes = train_y.size(1)
        num_labels = 3
        #metrics = Metrics()

        metrics = Metrics(chexpert_cols)
        print_freq = 5

        # Create DataLoader for training set, validation set, and test set
        train_dataset = TensorDataset(train_x, train_y, seq_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(val_x, val_y, seq_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(test_x, test_y, seq_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Instantiate the model and move it to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTM_Attn(input_size, hidden_size)
        model=model.to(device)

        # Define loss function and optimizer

        # Instantiate the custom loss function
        criterion = torch.nn.CrossEntropyLoss(weight=(weights).to(device))
        criterion2 = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Number of training epochs
        num_epochs = 500
        val_loss_box=[]
        train_loss_box=[]
        patience = 50  # Number of epochs to wait for improvement
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

            for batch_inputs, batch_targets, batch_seq_lens in train_loader:
                batch_inputs, batch_targets, batch_seq_lens = batch_inputs.to(device), batch_targets.to(device), batch_seq_lens.to(device)
                
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
                
                loss = criterion(temp1, temp2)
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
                for (batch_inputs, batch_targets, batch_seq_lens) in val_loader:
                    batch_inputs, batch_targets, batch_seq_lens = batch_inputs.to(device), batch_targets.to(device), batch_seq_lens.to(device)
                    outputs = model(batch_inputs, batch_seq_lens)
                    individual_outputs = torch.argmax(outputs, dim=2) # convert to nominals
                    sigmoid_matrices.append(outputs)
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

            metrics_dict = metrics.calculate_metrics()
            print(
                '\n * LOSS - {loss.avg:.3f}\n'.format(
                    loss=losses))
            pos_f1 = metrics_dict['Micro Positive F1']
            print('positive f1:',  pos_f1)
            if pos_f1 > best_f1:# and recall > best_recall:
                best_f1 = pos_f1
                #best_recall = recall
                # Save the model
                #torch.save(model.state_dict(), 'best_model_surr1.pth')
                print('Model saved! best f1: {:.4f}'.format(best_f1))
                current_patience = 0  # Reset patience counter
            else:
                current_patience += 1
                print('###################################################')
            if current_patience >= patience:
                print(f'Validation recall has not improved for {patience} epochs. Stopping training.')
                break

        # Testing phase

        criterion = torch.nn.CrossEntropyLoss()
        # Instantiate the model and move it to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loaded_model = AttentionLSTMClassifier(input_size, hidden_size,num_layers, output_size,3,dropout_rate=0.2)
        loaded_model = loaded_model.to(device)
        checkpoint = torch.load('best_model_surr1.pth')
        
        model.eval()
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
                outputs = model(batch_inputs, batch_seq_lens)
                individual_outputs = torch.argmax(outputs, dim=2)
                #sigmoid_outputs = torch.softmax(outputs, dim=1)
                sigmoid_matrices.append(outputs)
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
                correct_predictions = (batch_targets == individual_outputs).sum(dim=1).float()
                total_predictions = individual_outputs.shape[1]
                sample_accuracy = correct_predictions / total_predictions
                #print('sample wise accuracy: ', sample_accuracy)
                all_acc.append(sample_accuracy.cpu())
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
        metrics_dict = metrics.calculate_metrics()
        print(
            '\n * LOSS - {loss.avg:.3f}\n'.format(
                loss=losses))
        pos_f1 = metrics_dict['Micro Positive F1']
        print('all metrics: ', metrics_dict)
        print ( 'post_f1: ', pos_f1)
        print('all acc', flat_acc_list)
        print('all accuracy length: ', len(flat_acc_list))


        # code for second surrogate
        surr_2_x = torch.mean(test_x, dim=1)
        
        surr_2_y = flat_acc_list

        X = surr_2_x.cpu().detach().numpy()
        Y = surr_2_y.cpu().detach().numpy()

        

        train_x_=X[:int(len(X)*0.85)]
             
        train_y_=Y[:int(len(Y)*0.85)]

        test_x_=X[int(len(X)*0.85):] 
        test_y_=Y[int(len(Y)*0.85):]

        train_x_ = train_x_[:int(len(train_x_)*0.79)]
        train_y_ = train_y_[:int(len(train_y_)*0.79)]
        print(train_x_.shape)
        #train_x = torch.nn.functional.normalize(train_x)
        #train_y = torch.nn.functional.normalize(train_y)
        #weight_train = torch.nn.functional.normalize(weight_train)

        val_x_ = train_x_[int(len(train_x_)*0.79):]
        val_y_ = train_y_[int(len(train_y_)*0.79):]

        
        scaler_=MinMaxScaler()
        train_x_ = scaler_.fit_transform(train_x_)
        val_x_ = scaler_.transform(val_x_)
        test_x_ = scaler_.transform(test_x_)

        # save min max scalar
        # Extract the minimum and maximum values from the scaler
        min_value = scaler_.data_min_
        max_value = scaler_.data_max_

        # Convert the values to tensors
        min_tensor = torch.tensor(min_value)
        max_tensor = torch.tensor(max_value)

        # Save the tensors
        torch.save({'min': min_tensor, 'max': max_tensor}, 'min_max_scalar_params_embed_method.pt') #train_min_max_scalar_seq_40: see att_model.py. We take the seq_40 
        

        train_x_pt = torch.tensor(train_x_,dtype=torch.float) #pt: pytorch
        train_x_pt = train_x_pt/torch.norm(train_x_pt, dim = 1, keepdim= True)

        val_x_pt = torch.tensor(val_x_,dtype=torch.float) #pt: pytorch
        val_x_pt = val_x_pt/torch.norm(val_x_pt, dim = 1, keepdim= True)

        test_x_pt = torch.tensor(test_x_,dtype=torch.float)
        test_x_pt = test_x_pt/torch.norm(test_x_pt, dim = 1, keepdim= True)

        train_y_ = torch.tensor(train_y_)

        val_y_ = torch.tensor(val_y_)
        test_y_ = torch.tensor(test_y_)

        weight_train = torch.ones(len(train_x_))
        ############ Ridge regression ############

        # Define the hyperparameters
        input_size = train_x_pt.size(1)
        #weight_train = torch.ones(input_size)
        alpha = 0.05
        epochs = 300
        lr = 0.001
        ridge_loss=[]
        ridge_val_loss = []
        # Initialize the Ridge Regression model
        model_ridge = RidgeRegression(512, alpha).to(device)

        # Define the loss function
        loss_fn = CustomRidgeLoss(alpha)
        criterion1 = torch.nn.MSELoss()

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
            print('jjj',train_x_pt.size())
            outputs = model_ridge(torch.tensor(train_x_pt).to(device))
            print('outsize', outputs.size())
            print(train_y_.size())
            print(weight_train.size())
            #subtraction= (outputs.squeeze()-train_y_.to(device))
            #print('subtraction: ', subtraction)
            #print('subtraction size: ', subtraction.size())
            #loss = loss_fn(model_ridge,outputs.squeeze(), train_y_.to(device))  # Remove extra dimensions
            loss = loss_fn(model_ridge, weight_train.to(device), outputs.squeeze(),train_y_.to(device))
            ridge_loss.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            with torch.no_grad():
                val_out=model_ridge(torch.tensor(val_x_pt).to(device))
                val_loss=(criterion1(val_out.squeeze(), val_y_.reshape(-1).to(device)))
                ridge_val_loss.append(val_loss)
            #wandb.log({"train 3loss ridge reg": loss, "validation loss ridge reg": val_loss})
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, val_loss: {val_loss.item():.4f}')
            scheduler.step(val_loss)
            early_stopping(val_loss)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        # Retrieve the learned coefficients
        coefficients = model_ridge.linear.weight.cpu().detach().numpy()
        intercept = model_ridge.linear.bias.cpu().detach().numpy()
        # Evaluate the model on the test set
        with torch.no_grad():
            test_outputs = model_ridge(torch.tensor(test_x_pt).to(device))
            test_loss = (criterion1(test_outputs.squeeze(), test_y_.reshape(-1).to(device)))**0.5
        print('ridge gt: ',test_y_)
        print('ridge outputs: ', test_outputs)
        print('the rmse loss is: ', test_loss)
        rel_rse_num= torch.sum((test_outputs.view(-1)- test_y_.to(device))**2)
        rel_rse_den= torch.sum((test_y_.to(device)-torch.mean(test_y_).to(device))**2)
        rel_rse_ridge=rel_rse_num/rel_rse_den
        print("rel_rse_num: ", rel_rse_num)
        print("rel_rse_den: ", rel_rse_den)
        print("rel_rse_ridge: ", rel_rse_ridge)
        print("test variance: ", (torch.var(test_y_))**0.5)
        torch.save(model_ridge, 'surrogate/'+'ridge_second_surr_embed.pt')
        
        return log


