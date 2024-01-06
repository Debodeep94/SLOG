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
        #print(target_embedding_weights)
        torch.save(target_embedding_weights, 'target_embedding_weights.pth')



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
            img_ids, gs_box, weight_box_pred, embed_box_pred, seq_box_gt,samp_lps, gt_logps_box, seq_box_pred, embed_box_gt =[], [], [], [], [], [], [], [],[]
            count=0
            print('number of iterations required:', len(self.train_dataloader)) #for ground truth surrogate, we use train_dataloader
            for batch_idx, (images_id, images, reports_ids, reports_masks, seq_length) in tqdm(enumerate(self.train_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                seq_logps, seq, gs_logps = self.model(images, mode='sample')
                sequence_lengths_pred = torch.tensor([len(k) for k in seq]).to(torch.int64)
                #max_indices = np.argmax(seq_logprobs, axis=(1, 2))
                print('seq_logps_size: ', seq_logps.size())
                print('seq_gslogps_size: ', gs_logps.size())
                print('gt seq_size: ', reports_ids[:, 1:].size())
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
                weight_box_pred.extend(wts) #let's try with inverted blue1
                # Load the saved embedding weights directly into the new embedding layer
                target_embedding_weights= self.model.encoder_decoder.model.tgt_embed
                embedded_sentence = target_embedding_weights(reports_ids[:, 1:])
                embed_box_gt.extend(embedded_sentence)
                embedded_sentence_pred = target_embedding_weights(seq)
                embed_box_pred.extend(embedded_sentence_pred)
                gs_box.extend(gs_logps)
                # ground truth logps
                gt_logps = torch.nn.functional.one_hot(seq, gs_logps.size(2))
                gt_logps_box.extend(torch.log_softmax(gt_logps.to(torch.float32), dim=1))
                seq_box_gt.extend(seq_length)
                seq_box_pred.extend(sequence_lengths_pred)
                img_ids.extend(list(images_id))
                count+=1
                #print('latent_rep', latent_rep)
                #with open("file.txt", "w") as output_text:
                    #output_text.write(str(test_res))
                
                #if count == 50: # this needs to be user defined
                    #break
        # if we use gt embeddings, the weights are all 1
        weight_box_gt = [1]*len(weight_box_pred)            
        tensor_dict = dict(zip(img_ids, gs_box))
        tensor_dict_gt= dict(zip(img_ids, gt_logps_box))
        weight_dict_pred = dict(zip(img_ids, weight_box_pred))
        weight_dict_gt = dict(zip(img_ids, weight_box_gt))
        seq_dict_gt = dict(zip(img_ids,seq_box_gt))
        seq_dict_pred = dict(zip(img_ids,seq_box_pred))
        tensor_dict_lps = dict(zip(img_ids,samp_lps))
        embed_dict_gt = dict(zip(img_ids,embed_box_gt))
        embed_dict_pred = dict(zip(img_ids,embed_box_pred))
        path = '/home/debodeep.banerjee/R2Gen/surrogate_vectors/with_lps/'
        # Save the dictionary
        torch.save(weight_box_gt, path+'weight_box_gt.pt')
        torch.save(tensor_dict, path+'tensor_dict.pt')
        torch.save(tensor_dict_gt, path+'tensor_dict_gt.pt')
        torch.save(weight_dict_pred, path+'weight_dict_pred.pt')
        torch.save(weight_dict_gt, path+'weight_dict_gt.pt')
        torch.save(seq_dict_gt, path+'seq_dict_gt.pt')
        torch.save(seq_dict_pred, path+'seq_dict_pred.pt')
        
        #print(log)
        #return log
        
        

        