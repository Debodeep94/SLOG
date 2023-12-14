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



class Tester(BaseTester):

    def __init__(self, model, criterion, metric_ftns, args, test_dataloader, train_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        #self.surrogate_dataloader = surrogate_dataloader
    def test(self):
        
        #torch.manual_seed(42)
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        
        self.model.eval()
        with torch.no_grad():
            img_ids, gs_box, test_gts, test_res, weight_box, embed_box, seq_box, samp_lps = [], [], [], [], [], [], [],[]
            count=0
            print('number of iterations required:', len(self.train_dataloader)) #for ground truth surrogate, we use train_dataloader
            for batch_idx, (images_id, images, reports_ids, reports_masks, seq_length) in tqdm(enumerate(self.train_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                seq_logps, seq, gs_logps = self.model(images, mode='gumbel')
                #max_indices = np.argmax(seq_logprobs, axis=(1, 2))
                print('seq_logps_size: ', seq_logps.size())
                print('seq_gslogps_size: ', gs_logps.size())
                print('seq_size: ', seq.size())
                #latent= torch.split(latent_rep, split_size_or_sections=1, dim=0)
                #new_embedding_layer = torch.nn.Embedding(3216, 512)

                # Load the saved embedding weights directly into the new embedding layer
                saved_embedding_weights = torch.load('target_embedding_weights.pth')
                #new_embedding_layer=nn.Embedding.from_pretrained(saved_embedding_weights)
                target_embedding_weights= self.model.encoder_decoder.model.tgt_embed
                #new_embedding_layer.weight.data.copy_(saved_embedding_weights)
                #new_embedding_layer=new_embedding_layer.to(self.device)
                embedded_sentence = target_embedding_weights(reports_ids[:, 1:])
                print(embedded_sentence.size())
                #print('embedded_sentence: ', embedded_sentence)
                reports = self.model.tokenizer.decode_batch(seq.cpu().numpy())
                #print('log_ps: ', gs_logps)
                #print('log_ps_size:', gs_logps.size())
                #print('latent: ', latent)
                #print('latent_size:', latent.size())
                list_of_rep = [[item] for item in reports]
                
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                list_of_gt = [[item] for item in ground_truths]
                
                bleu_score, bleu_score_ind = self.metric_ftns({i: [gt] for i, gt in enumerate(ground_truths)},
                                        {i: [re] for i, re in enumerate(reports)})
                #print('bleu score', bleu_score_ind)
                wts=bleu_score_ind['BLEU_4']
                #print('wts: ', wts)
                weight_box.extend(wts) #let's try with inverted blue1
                
                #print('mean_weights ',mean_weights)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                gs_box.extend(gs_logps)
                samp_lps.extend(seq_logps)
                embed_box.extend(embedded_sentence)
                seq_box.extend(seq_length)
                img_ids.extend(list(images_id))
                count+=1
                #print('latent_rep', latent_rep)
                #with open("file.txt", "w") as output_text:
                    #output_text.write(str(test_res))
                
                #if count == 3: # this needs to be user defined
                    #break
        tensor_dict = dict(zip(img_ids, gs_box))
        weight_dict = dict(zip(img_ids, weight_box))
        embed_dict = dict(zip(img_ids,embed_box))
        seq_dict = dict(zip(img_ids,seq_box))
        tensor_dict_lps = dict(zip(img_ids,samp_lps))
        # Save the dictionary
        #torch.save(tensor_dict, 'tensor_dict_impressions_small.pt')
        #torch.save(weight_dict, 'all_weights_impressions_small.pt')
        #torch.save(embed_dict, 'all_embed_impressions_small.pt')
        #torch.save(test_res, 'pred_rep_only_find.pt')
        #print('length of dict: ', len(tensor_dict))
        #print('tensor dict: ',tensor_dict)
        test_met, test_met_ind = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
        log.update(**{'test_' + k: v for k, v in test_met.items()})
        print(log)
        
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
        lps_filt = {key: tensor_dict[key] for key in imp_filtered['study_id']}
        seq_filt = {key: seq_dict[key] for key in imp_filtered['study_id']}
        wt_filt = {key: weight_dict[key] for key in imp_filtered['study_id']}
        samp_lps_filt = {key: tensor_dict_lps[key] for key in imp_filtered['study_id']}

        
        tensors= list(lps_filt.values()) #tensor_dict: gumbel_softmax logps ; embed_dict: embedded outputs
        seq_lens=list(seq_filt.values())
        weight_values = list(wt_filt.values())
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

        wt_train = weight_values[:num_train_samples]
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
        wt_train = torch.tensor(wt_train)
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
        num_heads = 14
        #metrics = Metrics()

        metrics = Metrics(chexpert_cols)
        print_freq = 5

        # Create DataLoader for training set, validation set, and test set
        train_dataset = TensorDataset(train_x, train_y, seq_train, wt_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(val_x, val_y, seq_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(test_x, test_y, seq_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print('train_x_su1: ', train_x)
        # Instantiate the model and move it to GPU if available
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
                    #print('output: ', outputs.size())
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
                torch.save(model.state_dict(), 'best_model_surr1_tial.pth')
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
        loaded_model = LSTM_Attn(input_size, hidden_size,num_labels)#,num_layers, output_size,3,dropout_rate=0.2)
        loaded_model = loaded_model.to(device)
        checkpoint = torch.load('best_model_surr1_tial.pth')
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
        print('flat_acc_list: ',flat_acc_list)
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
        surr_2_x = test_x#torch.mean(test_x, dim=1)
        
        surr_2_y = flat_acc_list

        X = surr_2_x.cpu().detach().numpy()
        Y = surr_2_y.cpu().detach().numpy()
        seq_data = seq_test.cpu().detach().numpy()

        

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

        
        scaler_=MinMaxScaler()
        #train_x_ = scaler_.fit_transform(train_x_)
        #val_x_ = scaler_.transform(val_x_)
        #test_x_ = scaler_.transform(test_x_)

        # save min max scalar
        # Extract the minimum and maximum values from the scaler
        #min_value = scaler_.data_min_
        #max_value = scaler_.data_max_

        # Convert the values to tensors
        #min_tensor = torch.tensor(min_value)
        #max_tensor = torch.tensor(max_value)

        # Save the tensors
        #torch.save({'min': min_tensor, 'max': max_tensor}, 'min_max_scalar_params_samp_lps.pt') #train_min_max_scalar_seq_40: see att_model.py. We take the seq_40 
        

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
        # Create DataLoader for training set, validation set, and test set
        train_dataset = TensorDataset(train_x_pt, train_y_, seq_tr_)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(val_x_pt, val_y_, seq_val_)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(test_x_pt, test_y_, seq_test_)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = train_x_pt.size(-1)  # Vocabulary size
        hidden_size = 256   # Number of LSTM units
        num_layers = 8     # Number of LSTM layers
        output_size = train_y_.size(1)     # Number of output classes
        #print('output_size: ',output_size)
        batch_size = 256     # Batch size for training, validation, and testing
        learning_rate = 1e-5
        num_classes = train_y_.size(1)
        num_labels = 2
        num_heads = train_y_.size(1)
        weights = count_weights(train_y_)
        # Instantiate the model and move it to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_2 = LSTM_Attn(input_size, hidden_size, num_labels)#, num_classes, num_heads)
        model_2=model_2.to(device)

        # Accuracy surrogate
        # Instantiate the custom loss function
        criterion = torch.nn.CrossEntropyLoss(weight=(weights).to(device))#, reduction='none')
        criterion2 = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_2.parameters(), lr=learning_rate)

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
            model_2.train()
            total_loss = 0
            total_samples = 0

            for batch_inputs, batch_targets, batch_seq_lens in train_loader:
                batch_inputs, batch_targets, batch_seq_lens = batch_inputs.to(device), batch_targets.to(device), batch_seq_lens.to(device)#, batch_wt_train.to(device)
                
                # Modify target tensor to have the correct shape
                #batch_targets_modified = torch.zeros(batch_targets.size(0), batch_targets.size(1), 2).to(device)
                #batch_targets_modified.scatter_(2, batch_targets.unsqueeze(-1), 1)
                #print('batch_targets_modified:', batch_targets_modified)
                outputs = model_2(batch_inputs, batch_seq_lens)
                #specific_output=train_y[:,3]
                # Apply sigmoid activation and reshape the outputs
                #sigmoid_outputs = torch.softmax(outputs, dim=1)#.view(batch_targets.size(0), -1, 2)
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
            average_loss = total_loss / total_samples
            train_loss_box.append(average_loss)
            print('train loss: ',total_loss/(len(train_loader)))
            # Print training loss for the current epoch
            

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
                for (batch_inputs, batch_targets, batch_seq_lens) in val_loader:
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
            print('final_val_loss: ', final_val_loss)

            # When the batch job is done, check for the metrics
            # As we have two labels, things are easier now. 
            gt = torch.stack([item for sublist in full_batch for item in sublist])
            all_preds = torch.stack([item for sublist in pred_mat for item in sublist])
            #print(all_preds)
            f_micro = f1_score(gt.cpu(), all_preds.cpu(), average="micro")
            if f_micro > best_f1:# and recall > best_recall:
                best_f1 = f_micro
                #best_recall = recall
                # Save the model
                torch.save(model_2.state_dict(), 'best_model_surr2_tial.pth')
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
        checkpoint = torch.load('best_model_surr2_tial.pth')
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
        '''############ Ridge regression ############

        # Define the hyperparameters
        input_size = train_x_pt.size(1)
        #weight_train = torch.ones(input_size)
        alpha = 0.05
        epochs = 300
        lr = 0.001
        ridge_loss=[]
        ridge_val_loss = []
        # Initialize the Ridge Regression model
        model_ridge = RidgeRegression(input_size, alpha).to(device)

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
            outputs = model_ridge(train_x_pt.to(device))
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
                val_out=model_ridge(val_x_pt.to(device))
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
            test_outputs = model_ridge(test_x_pt.to(device))
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
        torch.save(model_ridge, 'surrogate/'+'ridge_second_surr_samp_logprobs.pt')'''
        
        return log


