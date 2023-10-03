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
#from torchviz import make_dot
#from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.utils import EarlyStopping
import csv

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
            img_ids, latent_rep_check, test_gts, test_res, weights = [], [], [], [], []
            count=0
            print('number of iterations required:', len(self.test_dataloader)) #for ground truth surrogate, we use train_dataloader
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.train_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                latent, seq = self.model(images, mode='sample')
                #max_indices = np.argmax(seq_logprobs, axis=(1, 2))
                #latent= torch.split(latent_rep, split_size_or_sections=1, dim=0)
                reports = self.model.tokenizer.decode_batch(seq.cpu().numpy())
                #print('reports: ', reports)
                list_of_rep = [[item] for item in reports]
                
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                list_of_gt = [[item] for item in ground_truths]
                
                bleu_score, bleu_score_ind = self.metric_ftns({i: [gt] for i, gt in enumerate(ground_truths)},
                                        {i: [re] for i, re in enumerate(reports)})
                #print('bleu score', bleu_score_ind)
                wts=bleu_score_ind['BLEU_4']
                #print('wts: ', wts)
                weights.extend(wts) #let's try with inverted blue1
                
                #print('mean_weights ',mean_weights)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                latent_rep_check.extend(latent)
                img_ids.extend(list(images_id))
                count+=1
                #print('latent_rep', latent_rep)
                #with open("file.txt", "w") as output_text:
                    #output_text.write(str(test_res))
                
                #if count == 2: # this needs to be user defined
                  #  break
        tensor_dict = dict(zip(img_ids, latent_rep_check))
        weight_dict = dict(zip(img_ids, weights))
        # Save the dictionary
        #torch.save(tensor_dict, 'tensor_dict_r1_more_data.pt')
        #torch.save(weight_dict, 'all_weights_only_find.pt')
        #torch.save(test_res, 'pred_rep_only_find.pt')
        #print('length of dict: ', len(tensor_dict))
        #print('tensor dict: ',tensor_dict)
        test_met, test_met_ind = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
        log.update(**{'test_' + k: v for k, v in test_met.items()})
        #print(len(latent_rep_check))
        #csv_file_path = "/home/debodeep.banerjee/R2Gen/csv_outputs/only_find/pred_reps_round1_more_data.csv"

        # Open the CSV file in write mode
        
        #with open(csv_file_path, mode='w', newline='') as csv_file:
            #csv_writer = csv.writer(csv_file)


        #    # Write each string in the list as a row in the CSV file
            #for string in test_res:
                # csv_writer.writerow([string])
        print(log)
        
        # Create the dataset with predictions and latent
        data_pred = {'study_id': img_ids, 'preds': test_res}
        data_pred = pd.DataFrame(data_pred)

        # Create the dataframe with the chexpert annotations
        chex = pd.read_csv('/home/debodeep.banerjee/chexpert-labeler/train_only_find0.0.csv')
        reports = list(chex['Reports'])
        chex = chex.iloc[:,1:]
        chex_cols=list(chex.columns)
        chex[chex_cols] = np.vectorize(convert)(chex[chex_cols])
        chex['info_score']=chex[chex_cols].sum(axis=1)
        info_score = list(chex['info_score'])
        # import the image ids

        with open(r"/home/debodeep.banerjee/R2Gen/data/mimic/only_find_new_split.json", 'r') as f:
            # Load the contents of the file into a variable
            data = json.load(f)
        image_id=[]
        for i in data['train']:
            image_id.append(i['study_id'])

        data = {'study_id': image_id, 'report': reports, 'info_score': info_score}
        chexpert = pd.DataFrame(data)

        # merge the two data
        merged_df = chexpert.merge(data_pred, on='study_id')
        # now calculate the BLEU scores
        response = list(merged_df['preds'])
        gt = list(merged_df['report'])

        #tr_weights=[]
        #for (m,n) in zip(gt, response):
            #   bleu, bleu_ind = self.metric_ftns({i: [gt] for i, gt in enumerate(m)},
                                            #{i: [re] for i, re in enumerate(n)})
            #  weight = bleu_ind['test_BLEU_4']
            # tr_weights.append(weight)
        
        #merged_df['weights'] = tr_weights
        merged_info_data = merged_df[['study_id', 'info_score']]
        #merged_weight_data = merged_df[['study_id', 'weights']]

        # convert info score data to a dict
        info_dict= merged_info_data.set_index('study_id').T.to_dict('list')
        # convert weight data to dict
        #weight_dict= merged_weight_data.set_index('study_id').T.to_dict('list')
            
        # creating data for surrogate model
        self.logger.info('Almost done....')
        chex=pd.read_csv(self.info_score_data)
        #chex['norm_info_score']=(np.array(chex['info_score'])-min(chex['info_score']))/(max(chex['info_score'])-min(chex['info_score']))
        chex=chex[['study_id', 'info_score']]
        print(chex.head())

        # Converting chexp to dictionary
        chex_dict= chex.set_index('study_id').T.to_dict('list')
        
        # Extract the common keys between chex and tensor dict
        
        # extract info scores aligned with latents
        score_bucket=[]
        for key1 in set(tensor_dict):
            #print(key1)
            for key2 in set(info_dict):
                if key1 == key2:
                    #print('Common key found for info sc. Storing the info score..')
                    info_sc = info_dict[key2]
                    score_bucket.append(info_sc)

        # extract weights aligned with latents
        weight_box=[]
        for key1 in set(tensor_dict):
            #print(key1)
            for key2 in set(weight_dict):
                if key1 == key2:
                    #print('Common key found for weights. Storing the info score..')
                    weight = weight_dict[key2]
                    weight_box.append(weight)
        
        print('length of score bucket is: ', len(score_bucket))
        print('length of weight bucket is: ', len(weight_box))
        tensors= list(tensor_dict.values())
        #tensors = [x / torch.norm(x) for x in tensors]
        
        train_x=tensors[:int(len(tensors)*0.85)]
        
        train_y=np.array(score_bucket[:int(len(score_bucket)*0.85)])

        test_x=tensors[int(len(tensors)*0.85):] 
        test_y=np.array(score_bucket[int(len(score_bucket)*0.85):])
        
        #weights = list(map(lambda i:[i], weights))
        weight_train= weight_box[:int(len(weights)*0.85)]
        weight_train = weight_train[:int(len(weight_train)*0.79)]
        
        train_x = train_x[:int(len(train_x)*0.79)]
        train_y = train_y[:int(len(train_y)*0.79)]

        #train_x = torch.nn.functional.normalize(train_x)
        #train_y = torch.nn.functional.normalize(train_y)
        #weight_train = torch.nn.functional.normalize(weight_train)

        val_x = train_x[int(len(train_x)*0.79):]
        val_y = train_y[int(len(train_y)*0.79):]

        test_list=[]
        for i in test_x:
            #i=i.cpu()
            new_list=i.tolist()
            test_list.append(new_list)
        test_list=np.array(test_list)

        val_list=[]
        for j in val_x:
            #i=i.cpu()
            new_list=j.tolist()
            val_list.append(new_list)
        val_list=np.array(val_list)

        train_list=[]
        for j in train_x:
            #i=i.cpu()
            new_list=j.tolist()
            train_list.append(new_list)
        train_list=np.array(train_list)
        #print('train_list: ', train_list)
        print(len(train_list))
        #minmaxscaler
        scaler_trainX = MinMaxScaler()
        scaler_weight = MinMaxScaler()
        scaler_trainY = MinMaxScaler()
        #scaler_test = MinMaxScaler()
        # transform data
        scaler_weight = MinMaxScaler()
        
        
        train_x = scaler_trainX.fit_transform(train_list.reshape(len(train_list),512))
        weight_train = scaler_weight.fit_transform(np.array(weight_train).reshape(-1, 1))
        #print('weight_train: ', weight_train)
        train_y = scaler_trainY.fit_transform(train_y)

        # Extract the minimum and maximum values from the scaler
        min_value = scaler_trainX.data_min_
        max_value = scaler_trainX.data_max_

        # Convert the values to tensors
        min_tensor = torch.tensor(min_value)
        max_tensor = torch.tensor(max_value)

        # Save the tensors
        #torch.save({'min': min_tensor, 'max': max_tensor}, 'train_min_max_scalar_only_find_updated_new.pt') #train_min_max_scalar_seq_40: see att_model.py. We take the seq_40 
        
        
        val_x = scaler_trainX.transform(val_list.reshape(len(val_list),512))
        val_y = scaler_trainY.transform(val_y.reshape(len(val_y),1))

        test_x = scaler_trainX.transform(test_list.reshape(len(test_list),512))
        test_y = scaler_trainY.transform(test_y.reshape(len(test_y),1))
        '''
        '''
        # save the data
        location= '/home/debodeep.banerjee/R2Gen/surrogate_vectors/'
        torch.save(train_x, location+'train_x_new_only_find_updated.pt')
        torch.save(train_y, location+'train_y_new_only_find_updated.pt')
        torch.save(val_x, location+'val_x_new_only_find_updated.pt')
        torch.save(val_y, location+'val_y_new_only_find_updated.pt')
        torch.save(test_x, location+'test_x_new_only_find_updated.pt')
        torch.save(test_y, location+'test_y_new_only_find_updated.pt')
        torch.save(weight_train, location+'weight_new_only_find_updated.pt')
        '''
        #len(train_y_pt)
        # Assume you have independent variables X and a dependent variable y
        '''
        train_x_pt = torch.tensor(train_x,dtype=torch.float) #pt: pytorch
        train_x_pt = train_x_pt/torch.norm(train_x_pt, dim = 1, keepdim= True)
        train_y_pt = torch.tensor(train_y, dtype=torch.float).reshape(-1,1) #pt:pytorch
        print('train_x_pt: ', train_x_pt)
        
        val_x_pt = torch.tensor(val_x,dtype=torch.float) #pt: pytorch
        val_x_pt = val_x_pt/torch.norm(val_x_pt, dim = 1, keepdim= True)
        val_y_pt = torch.tensor(val_y, dtype=torch.float).reshape(-1,1) #pt:pytorch

        test_x_pt = torch.tensor(test_x,dtype=torch.float)
        test_x_pt = test_x_pt/torch.norm(test_x_pt, dim = 1, keepdim= True)
        test_y_pt = torch.tensor(test_y, dtype=torch.float).reshape(-1,1)
        print('train_y_pt: ', test_y_pt)
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
        optimizer = torch.optim.Adam(model_surr.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=30, verbose=True)
        early_stopping = EarlyStopping(patience=50, delta=0)

        # Train the model
        num_epochs = 300
        train_loss = []
        validation_loss = []
        for epoch in range(num_epochs):
            y_hat = model_surr(train_x_pt)
            #print('yhat: ', y_hat)
            loss = criterion2(torch.tensor(weight_train), y_hat, train_y_pt)
            loss.backward()
            train_loss.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()
            
            y_hat_val = model_surr(val_x_pt)
            val_loss = criterion1(y_hat_val, val_y_pt)
            validation_loss.append(val_loss.item())
            #wandb.log({"train loss class reg": loss, "validation loss class reg": val_loss})
            if ((epoch+1)%10)==0:
                print('Epoch [{}/{}], Train Loss: {:.34f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), val_loss.item()))

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
        print('predicted', predicted)
        mse= criterion1(torch.tensor(predicted), test_y_pt)
        rel_rse=torch.sum((torch.tensor(predicted)- test_y_pt)**2)/torch.sum((test_y_pt-torch.mean(test_y_pt))**2)
        rmse=mse**0.5
        print('the rmse loss is ', rmse)
        print('Saving surrogate model...')
        torch.save(model_surr, 'surrogate/'+'surr_lin_reg_split_only_find_updated.pt')
        print('variance of test list: ', torch.std(test_y_pt))
        print('variance of pred list: ', np.std(np.array(predicted)))
        print('relative RSE: ', rel_rse)

        # Plot the curvesonly_find
        plt.style.use('ggplot')
        fig_surr=plt.figure(figsize=(16,9))
        plt.rcParams.update({'font.size': 30})
        plt.plot(train_loss, label= 'train loss', color='red', linewidth=2.0)
        plt.plot(validation_loss, label= 'validation loss', color='blue', linewidth=2.0)
        plt.legend()
        plt.title('Surrogate train on splitted training data')
        plt.savefig('plots/surr_lin_reg_split_only_find_updated.png')
        plt.close(fig_surr)

        ############ Ridge regression ############

        # Define the hyperparameters
        input_size = train_x_pt.size(1)
        alpha = 0.01
        epochs = 300
        lr = 0.001
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
            loss = loss_fn(model_ridge, torch.tensor(weight_train),outputs.squeeze(), train_y_pt)  # Remove extra dimensions
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
            #wandb.log({"train 3loss ridge reg": loss, "validation loss ridge reg": val_loss})
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
        print('ridge outputs: ', test_outputs)
        print('test_loss: ', test_loss)
        rel_rse_ridge=torch.sum((torch.tensor(test_outputs)- test_y_pt)**2)/torch.sum((test_y_pt-torch.mean(test_y_pt))**2)
        print("rel_rse_ridge: ", rel_rse_ridge)
        #torch.save(model_ridge, 'surrogate/'+'surr_ridge_reg_split_only_find_updated_new.pt')

        plt.style.use('ggplot')
        fig_ridge=plt.figure(figsize=(16,9))
        plt.rcParams.update({'font.size': 30})
        plt.plot(ridge_loss, label='train loss', color='red', linewidth=2.0)
        plt.plot(ridge_val_loss, label='validation loss', color='blue', linewidth=2.0)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend()
        plt.savefig('plots/surr_ridge_reg_split_only_find_updated.png')
        plt.close(fig_ridge)
        
        return log
