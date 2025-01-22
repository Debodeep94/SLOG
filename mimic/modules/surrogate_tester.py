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
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch
import itertools
import csv
import torch
from surrogate_module.rnn_surrogate import *
from surrogate_module.surrogate_utils import *
#from torchviz import make_dot
#from torchsummary import summary
import matplotlib.pyplot as plt
from CheXbert.src.label import *
import csv
from .ce_metrics import *

class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        #print(self.model[-1])
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        self.ann_path = args.ann_path
        self._load_checkpoint(args.load)
        self.surr_weight = args.surr_weight
        self.surrogate_model = args.surrogate_model
        self.min_max_scaler = args.min_max_scaler
        self.surrogate_model = args.surrogate_model
        self. surrogate_checkpoint = torch.load(self.surrogate_model)   
        self.surrogate = VisualSurrogate(2048,512,4, 256, 6, 14, 2, mode='CELoss')
        self.surrogate = self.surrogate.to(self.device)
        self.surrogate.load_state_dict(self.surrogate_checkpoint)
        for params in self.surrogate.parameters():
            params.requires_grad = False
        self.embed_weight = self.model.encoder_decoder.model.tgt_embed[0].lut.weight
        self.onehot_embedding = nn.Linear(self.embed_weight.shape[0], self.embed_weight.shape[1], bias=False).to(self.device)
        self.onehot_embedding.weight = torch.nn.Parameter(self.embed_weight.transpose(0,1))
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
        # print('checkpoint:', checkpoint)
        #surrogate_checkpoint= torch.load('/nfs/data_chaos/dbanerjee/my_data/R2Gen/surrogate/surr_model_sample_size_2500.pt')
        self.model.load_state_dict(checkpoint['state_dict'])


class SurrogateTester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, surrogate_dataloader):
        super(SurrogateTester, self).__init__(model, criterion, metric_ftns, args)
       # self.test_dataloader = test_dataloader
       # self.train_dataloader = train_dataloader
        self.surrogate_dataloader = surrogate_dataloader
    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            img_ids, latent_rep_check, test_gts, test_res, weights,pred_info_scores = [], [], [], [], [],[]
            count=0
            for batch_idx, (images_id, images, reports_ids, reports_masks,seq_length) in tqdm(enumerate(self.surrogate_dataloader)):
                images, reports_ids, reports_masks, seq_length = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), seq_length.to(self.device)
                # print('entering inference...')
                #print('image id: ', images_id)
                _, seq, seq_logps = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(seq.cpu().numpy())
                # print('report printing')
                sequence_lengths = torch.tensor([len(k) for k in seq]).to(torch.int64)
                embeddings= self.model.encoder_decoder.model.tgt_embed
                # we don't need this if we  are not using the embeddings
                img_ids.extend(images_id)
                # embedded_sentence_test = embeddings(seq)
                embedded_sentence_test = self.onehot_embedding(seq_logps)
                predicted = self.surrogate(images, embedded_sentence_test)
                # print('predicted: ', predicted)
                predicted = predicted[:,:,1]
                #print('predicted', predicted)
                reports = self.model.tokenizer.decode_batch(seq.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                #latent_rep_check.extend(latent)
                pred_info_scores.extend(predicted)
                count = count + 1 
                #if count == 5:
                    #break
                
        test_met, test_met_id = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
        log.update(**{'test_' + k: v for k, v in test_met.items()})
        concatenated_tensor = torch.cat(pred_info_scores)
        #print('info scores: ', torch.mean(concatenated_tensor,dim=1))
        print(len(pred_info_scores))
        print('mean info scores: ', torch.mean(concatenated_tensor))
        print('median info scores: ', torch.median(concatenated_tensor))
        print('test res: ', test_res)
        print('test gts: ', test_gts)
        #log.update(**{'mean info score':torch.mean(concatenated_tensor), 
                      #'median info score':torch.median(concatenated_tensor) })
        # save the file to csv for chexpert
        # Specify the CSV file path
        # Calculate the average length of sentences
        average_length = sum(len(sentence.split()) for sentence in test_res) / len(test_res)

        print(f"Average length of sentences: {average_length:.2f} words")
        torch.save(img_ids,"n/img_ids_"+str(self.surr_weight)+".pt")
        csv_file_path = "n/csv_outputs/imp_n_find/preds"+str(self.surr_weight)+".csv"

        # Open the CSV file in write mode
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write each string in the list as a row in the CSV file
            for string in test_res:
                csv_writer.writerow([string])
        
        csv_file_path = "n/csv_outputs/imp_n_find/output_R2Gen_imp_n_find_gt_round2"+str(self.surr_weight)+".csv"

        # Open the CSV file in write mode
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write each string in the list as a row in the CSV file
            for string in test_gts:
                csv_writer.writerow([string])
        csv_file_path1 = "n/csv_outputs/test/preds_test"+str(self.surr_weight)+".csv"
        csv_file_path2 = "n/csv_outputs/test/gts_test"+str(self.surr_weight)+".csv"
        # Open the CSV file in write mode for the preds
        column_name = "Report Impression"
        with open(csv_file_path1, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([column_name])
            # Write each string in the list as a row in the CSV file
            for string in test_res:
                csv_writer.writerow([string])
        
        with open(csv_file_path2, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([column_name])
            # Write each string in the list as a row in the CSV file
            for string in test_gts:
                csv_writer.writerow([string])


        chexbert_path='n/results/mimic_cxr/imp_n_find/chexbert.pth'
        output_path1="n/csv_outputs/test/chex_test_pred"+str(self.surr_weight)+".csv"
        output_path2="n/csv_outputs/test/chex_test_gts"+str(self.surr_weight)+".csv"
        column_name = "Report Impression"
        preds=chexbert_annotator(csv_file_path1,output_path1,chexbert_path)
        gts=chexbert_annotator(csv_file_path2,output_path2,chexbert_path)
        print(preds)
        prec_dict, recall_dict, f1_dict=ce_metrics(output_path1,output_path2, 1)
        print(prec_dict)
        print(recall_dict)
        print(f1_dict)
        pos_f1=f1_dict['F1_MICRO']
        log.update(**{'val_' +'POSITIVE_F1': pos_f1})
        print(log)
        # this is for chexpert-tool
        # column_name = "Report Impression"
        with open(csv_file_path1, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # csv_writer.writerow([column_name])
            # Write each string in the list as a row in the CSV file
            for string in test_res:
                csv_writer.writerow([string])
        return log
