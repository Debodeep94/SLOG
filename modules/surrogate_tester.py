import logging
import os
from abc import abstractmethod
import json
import pandas as pd
import ast
import cv2
import torch
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
#from torchviz import make_dot
#from torchsummary import summary
import matplotlib.pyplot as plt
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
        #print('checkpoint:', checkpoint)
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
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.surrogate_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                print('entering inference...')
                #print('image id: ', images_id)
                output, latent, score = self.model(images, mode='surrogate')
                #print('latent: ', latent)
                #print('latent size: ', latent.size())
                pred_info_scores.extend(score)
                latent= torch.split(latent, split_size_or_sections=1, dim=0)
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                print('report printing')
                #print('reports: ', reports)
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                latent_rep_check.extend(latent)
                
        test_met, test_met_id = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
        log.update(**{'test_' + k: v for k, v in test_met.items()})
        concatenated_tensor = torch.cat(pred_info_scores)
        print('info scores: ', pred_info_scores)
        print('mean info scores: ', torch.mean(concatenated_tensor))
        print('median info scores: ', torch.median(concatenated_tensor))
        print('test res: ', test_res)
        print('test gts: ', test_gts)
        log.update(**{'mean info score':torch.mean(concatenated_tensor), 
                      'median info score':torch.median(concatenated_tensor) })
        # save the file to csv for chexpert
        # Specify the CSV file path
        '''
        csv_file_path = "/home/debodeep.banerjee/R2Gen/csv_outputs/only_find/chex/no_oracle/output_"+str(self.surr_weight)+".csv"

        # Open the CSV file in write mode
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write each string in the list as a row in the CSV file
            for string in test_res:
                csv_writer.writerow([string])
        '''
        print(log)
        return log

