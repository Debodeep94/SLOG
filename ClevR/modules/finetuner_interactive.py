import os
from abc import abstractmethod
#from apex import amp
import time
import torch
import pandas as pd
from numpy import inf
import logging
import os
from abc import abstractmethod
import json
#import ast
import cv2
import torch
import seaborn as sns
import numpy as np
from modules.utils import generate_heatmap
from surrogate import SurrogateModel, SurrogateLoss, CustomRidgeLoss, RidgeRegression
from surrogate import SurrogateLoss
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import itertools
from pycocoevalcap.bleu.bleu import Bleu
#from torchviz import make_dot
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from modules.utils import cycle
from torch.autograd import Variable
import numpy as np
from modules.utils import generate_heatmap, surrogate_regression, surrogate_split
from surrogate import SurrogateModel, SurrogateLoss, CustomRidgeLoss, RidgeRegression
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import itertools
import random
from modules.loss import jsd_loss
from torchviz import make_dot
#from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.utils import EarlyStopping, torch_minmax
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import wandb
import csv
import importlib
import sys
import logging
from tqdm import tqdm

from constants import *
#sys.path.append('/home/debodeep.banerjee/R2Gen/label.py')
print('GPU availability: ', torch.cuda.is_available())

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        #torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
            self.model.to(self.device)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        #self.info_score_data= args.info_score_data

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        self.batch_size = args.batch_size
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.ce_weight= args.llm_weight
        self.surr_weight = args.surr_weight
        

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}#,
                              #'test': {self.mnt_metric_test: self.mnt_best}}
        #self.writer = SummaryWriter(log_dir='logs')

        self.surrogate_model = args.surrogate_model
        self.min_max_scaler = args.min_max_scaler
    print('DID YOU PUT A CORRECT LAMBDA VALUE TO THE WANDB CONFIG?????')
    wandb.init(
    # set the wandb project where this run will be logged
    project="Finetune_Lamda_1",
    
    # track hyperparameters and run metadata
    config={
    "architecture": "transformer",
    "dataset": "mimic train 10845",
    "epochs": 100,
        },
    settings=wandb.Settings(_service_wait=300)
    )
    @abstractmethod
    
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        loss=[]
        val_info_scores=[]
        count_epochs=0
        #directory_to_add = "/home/debodeep.banerjee/chexpert-labeler/NegBio"
        #os.environ['PYTHONPATH'] = f"{directory_to_add}:{os.environ.get('PYTHONPATH', '')}"
        
        for epoch in range(self.start_epoch, self.epochs + 1): 
            result, train_loss, val_info_score = self._train_epoch(epoch)
            #flat_list = [item for sublist in val_info_score.tolist() for item in sublist] # convert the list of lists into a single list
            val_info_scores.append(val_info_score)
            print(result)
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)
            loss.append(train_loss.item())
            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            count_epochs+=1
        print(len(val_info_scores))
        #print(val_info_scores)
        # plot information score boxplot
        fig_box = plt.figure()
        plt.boxplot(val_info_scores)
        plt.title('Box plot of information scores of validation data over epochs')
        plt.savefig('/home/debodeep.banerjee/R2Gen/plots/only_find/interactive/'+str(self.ce_weight).replace('.', '_')+'_SR'+str(self.surr_weight).replace('.', '_')+'.png')
        plt.close(fig_box)
        # plot train list
        fig=plt.figure()
        plt.plot(loss)
        plt.title('Train+Finetune')
        plt.savefig('/home/debodeep.banerjee/R2Gen/plots/only_find/interactive/'+str(self.ce_weight).replace('.', '_')+'_SR'+str(self.surr_weight).replace('.', '_')+'.png')
        plt.close(fig)

        return log
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        #self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        #self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        #self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        #record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        print('device: ', device)
        print('list: ', list_ids)
        return device, list_ids
    
    

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'test_trial_mimic_new_split_CE_surr_'+str(self.ce_weight).replace('.', '_')+'_SR'+str(self.surr_weight).replace('.', '_')+'.pth') # startid=0; num_ele=20
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'test_trial_best_new_split_CE_surr_'+str(self.ce_weight).replace('.', '_')+'_sr'+str(self.surr_weight).replace('.', '_')+'.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best2.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1   # checkpoint['epoch'] + 1 just to check how it goes. 
        print("Resuming from epoch: {} ...".format(self.start_epoch))
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        print(checkpoint)
        print(checkpoint['epoch'])
        #surrogate_checkpoint= torch.load('/nfs/data_chaos/dbanerjee/my_data/R2Gen/surrogate/surr_model_sample_size_2500.pt')
        self.model.load_state_dict(checkpoint['state_dict'])


        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        '''improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)'''

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        '''print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))'''


class FineTuner(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(FineTuner, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
    def _train_epoch(self, epoch):
        log_dir = "logs/"   # Replace with the directory where you want to save the logs
        
        print('entering train_epoch')
        
        # Create a new iterator that starts from the specified index and returns the specified number of elements
        
        #latent_rep_check, images_ids, test_res, test_gts, weights=[], [], [], [], []
        '''train_dataloader_cycle = itertools.cycle(self.train_dataloader)
        test_dataloader_cycle = itertools.cycle(self.test_dataloader)

        train_dataloader_size = len(self.train_dataloader)
        test_dataloader_size = len(self.test_dataloader)
        max_dataloader_size = max(train_dataloader_size, test_dataloader_size)'''
        loss_llm = 0
        hybrid=0
        tot_surr_loss=0
        count = 0
        
        # Converting all the parameters to float16
        
        self.model.train()
        total_length = len(self.test_dataloader)

        print('total_length: ', total_length)
        print('------------------------------------------------------------------------------------------------')
        print('surrogate weight', self.surr_weight)
        print('LLM weight', self.ce_weight)
        print('------------------------------------------------------------------------------------------------')
        start= time.time()
        for batch_idx_llm, ((images_id, images, reports_ids, reports_masks), 
                           (images_id_surr_ft, images_surr_ft, reports_ids_surr_ft, reports_masks_surr_ft)) in tqdm(enumerate(zip(self.train_dataloader, cycle(self.test_dataloader)))): #because ft is less than train here
        #for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.train_dataloader)):
            
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
            
            images_ft, reports_ids_ft, reports_masks_ft = images_surr_ft.to(self.device), reports_ids_surr_ft.to(self.device), reports_masks_surr_ft.to(self.device)
            
            print('Entering Training Loop')
            print('Running training mode')
            
            output = self.model(images, reports_ids, mode='train')

            loss_lang = self.criterion(output, reports_ids, reports_masks)
            
            #jsd = jsd_loss(output+epsilon, output_ref+epsilon)

            #print('JSD loss', jsd)
            print('Cross entropy loss', loss_lang)
            #print(loss_lang.item())
            loss_llm+=loss_lang.item()#.append(loss_lang.item())
            print('cumulative loss: ', loss_llm)
            
            print('Running sample mode')
            #self.model.eval()
            output_surr, latent = self.model(images,reports_ids, mode='sample')
            #print(latent.dtype)
            output_surr_ft, lat_vec_ft = self.model(images_ft,reports_ids_ft, mode='sample')
            loaded_data = torch.load(self.min_max_scaler)

            # Retrieve the min and max tensors
            min_tensor = loaded_data['min']
            max_tensor = loaded_data['max']
            min_tensor = min_tensor.to(self.device)
            max_tensor = max_tensor.to(self.device)
            test_x_ft = lat_vec_ft.sub(min_tensor).div(max_tensor - min_tensor)
            test_x_ft = test_x_ft.to(torch.float32)
            test_x_ft = test_x_ft/torch.norm(lat_vec_ft,dim=1, keepdim=True)
        
            test_x_tr = latent.sub(min_tensor).div(max_tensor - min_tensor) 
            test_x_tr = test_x_tr.to(torch.float32)
            test_x_tr = test_x_tr/torch.norm(latent,dim=1, keepdim=True)
            #test_x_tr = latent# torch_minmax(latent, self.device)
            print('surrogate model: ', self.surrogate_model)
            print('min max scalar : ',self.min_max_scaler)
            surrogate = torch.load(self.surrogate_model)#self.surrogate_model)   
            surrogate = surrogate.to(self.device)
            for params in surrogate.parameters():
                params.requires_grad = False
            #regression_params = surrogate.parameters()
             
            predicted_ft = surrogate(test_x_ft)
            #print('predicted ft: ', predicted_ft)
            surr_ft_score = torch.mean(predicted_ft)
            #surr_ft_score = surr_ft_score.clone().requires_grad_(True)
            print('surrogate finetune score: ', surr_ft_score)

            predicted_tr = surrogate(test_x_tr)
            #print('predicted train: ', predicted_tr)
            surr_tr_score = torch.mean(predicted_tr)
            #surr_tr_score = surr_tr_score.clone().requires_grad_(True)
            print('surr train score: ', surr_tr_score)
            print('ce loss: ', loss_lang)
            surrogate_loss = surr_ft_score+surr_tr_score
            tot_surr_loss+=surrogate_loss.item()
            print('surrogate loss: ', surrogate_loss)
            
            surrogate_loss = self.surr_weight*surrogate_loss
            ce_loss = self.ce_weight*loss_lang
            hdm_loss =  ce_loss-surrogate_loss
            #hdm_loss =  hdm_loss.clone().requires_grad_(True)
            hybrid+=hdm_loss.item()
            #hdm_loss = -surrogate_loss
            print('hdm_loss: ', hdm_loss)
            self.optimizer.zero_grad()
            hdm_loss.backward()
            print(hdm_loss.grad_fn)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            print('all done')
            count = count+1
            
            

        log = {'total cross entropy loss: ': loss_llm/ count}
        log.update(**{'total hdm loss: ': hybrid/ count})

        wandb.log({"Cross entropy loss": loss_llm/ count,
                        'surrogate loss': tot_surr_loss/ count,
                        'hdm_loss': hybrid/ count
                        })
        print(len(self.train_dataloader))
        print('log: ', log)

        # Entering validation

        print('entering validation')
        with torch.no_grad():
            count_val=0
            val_gts, val_res, val_inf_pred = [], [],[]
            for batch_idx_val, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output,_, predicted_ft = self.model(images, mode='surrogate')
                #print(predicted_ft)
                #output_val,_ = self.model(images, mode='sample')
                predicted_ft = [item for sublist in predicted_ft.tolist() for item in sublist]
                val_inf_pred.extend(predicted_ft)
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                #count_val=count_val+1

            print('pred text', val_res)
            
            print('grund truth text', val_gts)
            val_met, val_met_ind = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                    {i: [re] for i, re in enumerate(val_res)})
            #print(val_met)                       
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        #torch.cuda.empty_cache()
        print('len val info sc:', len(val_inf_pred))
        mean_inf_sc = np.mean(val_inf_pred)
        wandb.log({'validation info score':mean_inf_sc})
        log.update(**{'val info score: ': mean_inf_sc})
        return log, hdm_loss, val_inf_pred #this should be replaced by hdm_loss
