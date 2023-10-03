import os
from abc import abstractmethod
import time
import torch
import pandas as pd
from numpy import inf
import logging
import os
from abc import abstractmethod
import json
import cv2
import torch
import numpy as np
from modules.utils import generate_heatmap
from surrogate import SurrogateModel, SurrogateLoss, CustomRidgeLoss, RidgeRegression
from surrogate import SurrogateLoss
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import itertools
from pycocoevalcap.bleu.bleu import Bleu
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from modules.utils import cycle
from torch.autograd import Variable
from torch.utils.data import DataLoader

print('GPU availability: ', torch.cuda.is_available())

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            print('Data parallel enabled!')

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.info_score_data = args.info_score_data

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.ce_weight = args.llm_weight
        self.surr_weight = args.surr_weight

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        loss = []
        val_info_scores = []
        count_epochs = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            result, train_loss, val_info_score = self._train_epoch(epoch)
            val_info_scores.append(val_info_score)
            print(result)

            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)
            loss.append(train_loss.item())

            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off':
                try:
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
            count_epochs += 1

        fig_box = plt.figure()
        plt.boxplot(val_info_scores)
        plt.title('Box plot of information scores of validation data over epochs')
        plt.savefig('plots/boxFT_ridge_new_split_2' + str(self.ce_weight) + '_sr' + str(self.surr_weight) + '.png')
        plt.close(fig_box)

        fig = plt.figure()
        plt.plot(loss)
        plt.title('Train+Finetune')
        plt.savefig('plots/train&ft_ridge_mimic_new_split_2' + str(self.ce_weight) + '_sr' + str(self.surr_weight) + '.png')
        plt.close(fig)

        return log

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
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
        filename = os.path.join(self.checkpoint_dir, 'mimic_new_split_2' + str(self.ce_weight) + '_sr' + str(
            self.surr_weight) + '.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_new_split_2' + str(self.ce_weight) + '_sr' + str(
                self.surr_weight) + '.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best2.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        print("Resuming from epoch: {} ...".format(self.start_epoch))
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

class FineTuner(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(FineTuner, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        log_dir = "logs/"

        loss_llm = 0
        hybrid = 0
        count = 0
        self.model.train()
        total_length = len(self.train_dataloader)
        total_ft_length = len(self.test_dataloader)
        print('total length: ', total_length)
        print('total FT length: ', total_ft_length)

        start = time.time()
        print('dataloading started at: ', start)
        self.model = self.model.to(self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            print("Model is spread across all GPUs.")
        else:
            print("Model is not spread across all GPUs.")
        for batch_idx_llm, ((images_id, images, reports_ids, reports_masks),
                            (images_id_surr_ft, images_surr_ft, reports_ids_surr_ft, reports_masks_surr_ft)) in enumerate(zip(
            cycle(self.train_dataloader), self.test_dataloader)):

            end = time.time()
            start_iter = time.time()
            print('total time taken to dataloading: ', end - start)

            images =images.to( self.device)
            reports_ids = reports_ids.to( self.device) 
            reports_masks = reports_masks.to( self.device)
            
            images_surr_ft = images_surr_ft.to( self.device)
            reports_ids_surr_ft = reports_ids_surr_ft.to( self.device)
            reports_masks_surr_ft = reports_masks_surr_ft.to( self.device)

            print('------------------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------------------')

            print(images_surr_ft.device)
            print(reports_ids_surr_ft.device)
            print(reports_masks_surr_ft.device)

            print(images.device)
            print(reports_ids.device)
            print(reports_masks.device)

            print('------------------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------------------')

            print('Entering Training Loop')
            print('Running training mode')
            output, _ = self.model(images, reports_ids, mode='train')
            print('output done')

            loss_lang = self.criterion(output, reports_ids, reports_masks)
            print('Cross entropy loss', loss_lang)
            loss_llm += loss_lang.item()
            print('cumulative loss: ', loss_llm)

            print('Running sample mode')
            with torch.no_grad():
                output_surr, latent = self.model(images, mode='sample')

            output_surr_ft, lat_vec_ft = self.model(images_surr_ft, mode='sample')

            print('surrogate loss: ', (surr_ft_score + surr_ft_score))
            print('ce loss: ', loss_lang)
            surrogate_loss = surr_ft_score + surr_ft_score
            surrogate_loss = surrogate_loss.to(loss_lang.device)
            hdm_loss = torch.add(self.ce_weight * loss_lang, (-self.surr_weight) * surrogate_loss)
            hybrid += hdm_loss.item()
            print('hdm_loss: ', hdm_loss)

            self.optimizer.zero_grad()
            loss_lang.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            print('all done')

        log = {'total cross entropy loss: ': loss_llm / count}
        log.update(**{'total hdm loss: ': hybrid / count})
        print(len(self.train_dataloader))
        print('log: ', log)

        self.model.eval()
        print('entering validation ...')
        end_iter = time.time()
        print('time taken for one iteration: ', (end_iter - start_iter) / 60)

        with torch.no_grad():
            val_gts, val_res, val_inf_pred = [], [], []
            for batch_idx_val, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output, _, predicted_ft = self.model(images, mode='surrogate')
                predicted_ft = [item for sublist in predicted_ft.tolist() for item in sublist]
                val_inf_pred.extend(predicted_ft)
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met, val_met_ind = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                                    {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        print('len val info sc:', len(val_inf_pred))
        return log, hdm_loss, val_inf_pred
