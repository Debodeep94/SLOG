import os
from abc import abstractmethod
#from apex import amp
import time
import torch
import torch.nn as nn
import pandas as pd
from numpy import inf
import logging
import os
from abc import abstractmethod
import json
#import ast
import cv2
import torch
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import itertools
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from modules.utils import cycle
from torch.autograd import Variable
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
import random
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
#import wandb
from surrogate_module.surrogate_utils import *
from surrogate_module.rnn_surrogate import *

print('GPU availability: ', torch.cuda.is_available())
class BaseTrainer(object):
    def __init__(self, model, criterion,surr_criterion, metric_ftns, optimizer, args):
        self.args = args
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        #torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
            self.model.to(self.device)

        self.criterion = criterion
        self.surr_criterion = surr_criterion
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
        if args.load is not None:
            self._load_checkpoint(args.load)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}#,
                              #'test': {self.mnt_metric_test: self.mnt_best}}
        #self.writer = SummaryWriter(log_dir='logs')

        self.surrogate_model = args.surrogate_model
    #print('DID YOU PUT A CORRECT LAMBDA VALUE TO THE WANDB CONFIG?????')
    
    '''wandb.init(
        # set the wandb project where this run will be logged
        project="Finetune_Lamda_0",
        
        # track hyperparameters and run metadata
        config={
        "architecture": "transformer",
        "dataset": "mimic train 10845",
        "epochs": 100,
            }
        )'''
    @abstractmethod
    
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        loss=[]
        val_info_scores=[]
        count_epochs=0
        for epoch in range(self.start_epoch, self.epochs + 1):
            #if epoch!=1:
                #self.surrogate_model = 'results/workshop/'+'surrogate_weights_ratio50.pt'
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
        #plt.boxplot(val_info_scores)
        #plt.title('Box plot of information scores of validation data over epochs')
        #plt.savefig('/home/debodeep.banerjee/R2Gen/plots/only_find/'+str(self.ce_weight).replace('.', '_')+'_SR'+str(self.surr_weight).replace('.', '_')+'.png')
        #plt.close(fig_box)
        # plot train list
        fig=plt.figure()
        plt.plot(loss)
        plt.title('Train+Finetune')
        #plt.savefig('/home/debodeep.banerjee/R2Gen/only_find/'+str(self.ce_weight).replace('.', '_')+'_SR'+str(self.surr_weight).replace('.', '_')+'.png')
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
        filename = os.path.join(self.checkpoint_dir, 'finetuned_checkpoint_CE_surr_swap_'+str(self.ce_weight).replace('.', '_')+'_SR'+str(self.surr_weight).replace('.', '_')+str(epoch)+'.pth') # startid=0; num_ele=20
        #torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'finetuned_best_CE_surr_swap_'+str(self.ce_weight).replace('.', '_')+'_sr'+str(self.surr_weight).replace('.', '_')+str(epoch)+'.pth')
            #torch.save(state, best_path)
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
        #surrogate_checkpoint= torch.load('/nfs/data_chaos/dbanerjee/my_data/R2Gen/surrogate/surr_model_sample_size_2500.pt')
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded. Initiating finetuning from epoch {}".format(self.start_epoch))
        self.optimizer.load_state_dict(checkpoint['optimizer'])

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
    def __init__(self, model, criterion,surr_criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader):
        super(FineTuner, self).__init__(model, criterion,surr_criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        #self.test_dataloader = test_dataloader
        
    def _train_epoch(self, epoch):
        log_dir = "logs/"   # Replace with the directory where you want to save the logs
        
        print('entering train_epoch')
        loss_llm = 0
        hybrid=0
        tot_surr_loss=0
        count = 0

        # Converting all the parameters to float16
        
        self.model.train()
        total_length = len(self.train_dataloader)

        print('total_length: ', total_length)
        print('------------------------------------------------------------------------------------------------')
        print('surrogate weight', self.surr_weight)
        print('LLM weight', self.ce_weight)
        print('------------------------------------------------------------------------------------------------')
        for batch_idx, (images_id, images, reports_ids, reports_masks, seq_length, chex_targets) in tqdm(enumerate(self.train_dataloader)):
            
            images, reports_ids, reports_masks, chex_targets = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device), chex_targets.to(self.device)
            
            #images_ft, reports_ids_ft, reports_masks_ft = images_surr_ft.to(self.device), reports_ids_surr_ft.to(self.device), reports_masks_surr_ft.to(self.device)
            
            print('Entering Training Loop')
            print('Running training mode')
            
            output = self.model(images, reports_ids, mode='train')

            loss_lang = self.criterion(output, reports_ids, reports_masks)
            print('Cross entropy loss', loss_lang)
            #print(loss_lang.item())
            loss_llm+=loss_lang.item()#.append(loss_lang.item())
            print('cumulative loss: ', loss_llm)
            
            print('Running sample mode')
            
            #latent, output_surr = self.model(images,reports_ids, mode='sample')
            latent, seq, gs_logps_tr = self.model(images, mode='gumbel')
            #lat_vec_ft, seq_ft, gs_logps_ft = self.model(images_ft, mode='sample')
            for params in self.model.visual_extractor.parameters():
                params.requires_grad=False

            embeddings= self.model.encoder_decoder.model.tgt_embed

            # we don't need this if we  are not using the embeddings
            embedded_sentence_tr = embeddings(seq)
            #embedded_sentence_ft = embeddings(seq_ft)

            # sequence lengths
            sequence_lengths_tr = torch.tensor([len(k) for k in seq]).to(torch.int64)
            #sequence_lengths_ft = torch.tensor([len(k_ft) for k_ft in seq_ft]).to(torch.int64)
            
            print(self.surrogate_model)
            
            surrogate_checkpoint = torch.load(self.surrogate_model)   
            surrogate = LSTM_Attn(embedded_sentence_tr.size(-1),1024, 3)#,num_layers, output_size,3,dropout_rate=0.2)
            surrogate = surrogate.to(self.device)
            surrogate.load_state_dict(surrogate_checkpoint)
            
            for params in surrogate.parameters():
                params.requires_grad = False
            
            #predicted_ft = surrogate(embedded_sentence_ft,sequence_lengths_ft)
            #predicted_ft = predicted_ft[:,:,1]
            #print('predicted ft: ', predicted_ft)
            predicted_tr = surrogate(embedded_sentence_tr,sequence_lengths_tr)
            #print(predicted_tr)
            surrogate_loss = self.surr_criterion(predicted_tr,chex_targets)
            print('ce loss: ', loss_lang)
            #surrogate_loss = surr_tr_score
            tot_surr_loss+=surrogate_loss.item()
            print('surrogate loss: ', surrogate_loss.item())
            
            surrogate_loss = self.surr_weight*surrogate_loss
            ce_loss = self.ce_weight*loss_lang
            hdm_loss =  ce_loss+surrogate_loss
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
            print('------------------------------------------------------------------------------------------------')
            count+=1
            #if count==100:
                #break  
            
        
        log = {'total cross entropy loss: ': loss_llm/ count}
        log.update(**{'total hdm loss: ': hybrid/ count})

        '''wandb.log({"Cross entropy loss": loss_llm/ count,
                        'surrogate loss': tot_surr_loss/ count,
                        'hdm_loss': hybrid/ count
                        })'''
        print(len(self.train_dataloader))
        print('log: ', log)
        
        # Create a new iterator that starts from the specified index and returns the specified number of elements
        self.model.eval()
        print('entering validation ...')
        #self.model.eval()
        metrics_val = Metrics(['Atelectasis', 'Cardiomegaly', 'Consolidation', 
                               'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 
                               'Lung Lesion', 'Lung Opacity', 'No Finding', 
                               'Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                               'Pneumothorax', 'Support Devices'])
        with torch.no_grad():
            count_val=0
            tot_val_surr_loss=0
            val_gts, val_res, val_inf_pred = [], [],[]
            for batch_idx_val, (images_id, images, reports_ids, reports_masks, seq_length, chex_targets) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks, chex_targets = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), chex_targets.to(self.device)
                latent_val, seq_val, gs_logps_val = self.model(images, mode='sample')
                sequence_lengths_val = torch.tensor([len(k) for k in seq_val]).to(torch.int64)
                embeddings= self.model.encoder_decoder.model.tgt_embed

                # we don't need this if we  are not using the embeddings
                embedded_sentence_val = embeddings(seq_val)
                surrogate_checkpoint = torch.load(self.surrogate_model)   
                surrogate = LSTM_Attn(embedded_sentence_val.size(-1), 1024, 3)#,num_layers, output_size,3,dropout_rate=0.2)
                surrogate = surrogate.to(self.device)
                surrogate.load_state_dict(surrogate_checkpoint)
                predicted_val = surrogate(embedded_sentence_val,sequence_lengths_val)
                surrogate_loss = self.surr_criterion(predicted_val,chex_targets, weights='no')
                print('ce loss: ', loss_lang)
                #surrogate_loss = surr_tr_score
                tot_val_surr_loss+=surrogate_loss.item()
                metrics_val.update(predicted_val.to('cpu'), chex_targets.to('cpu'))
                reports = self.model.tokenizer.decode_batch(seq_val.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                #count_val=count_val+1
            #concatenated_tensor = torch.cat(val_inf_pred)
            print('pred text', val_res)
            
            print('grund truth text', val_gts)
            val_met, val_met_ind = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                    {i: [re] for i, re in enumerate(val_res)})
            #print(val_met)     
            metrics_dict_val = metrics_val.calculate_metrics() 
            pos_f1_gt = metrics_dict_val['Micro Positive F1']
            
            print('positive f1 gt:',  pos_f1_gt)
            surrogate_loss = tot_val_surr_loss/len(self.val_dataloader)   
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        #torch.cuda.empty_cache()
        #print('len val info sc:', len(predicted_val))
        #mean_inf_sc = torch.mean(concatenated_tensor)
        #wandb.log({'validation info score':mean_inf_sc})
        print('SURROGATE LOSS: ',surrogate_loss)
        log.update(**{'SURROGATE_SCORE: ': pos_f1_gt})
        #log.update(**{'SURROGATE_LOSS: ', surrogate_loss})

        self.lr_scheduler.step()
        return log, hdm_loss, val_inf_pred #this should be replaced by hdm_loss
