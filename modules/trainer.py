import os
from abc import abstractmethod
import logging
import time
import torch
import pandas as pd
from numpy import inf
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DataParallel as DP
from modules.utils import check_tensor_device
from tqdm import tqdm
#for multi GPU training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp



class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):

        self.args = args
        #torch.cuda.set_device(0)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        #self.fixed_device = torch.device('cuda:0')
        self.model = model.to(self.device)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = DP(self.model,device_ids=device_ids)
        
        # print the device of each parameter
        #for name, param in self.model.named_parameters():
         #   print(name, param.device)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        #self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}#,
                              #'test': {self.mnt_metric_test: self.mnt_best}}
    
        

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        loss=[]
        count_epoch=0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, lang_loss = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)
            loss.append(lang_loss)
            
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
            count_epoch+=1
        fig= plt.plot(range(count_epoch+1), loss)
        fig= plt.title('Train Loss')
        plt.savefig('/home/debodeep.banerjee/R2GenCMN/plots/train_imp_n_find.png')
        return log
        self._print_best()
        self._print_best_to_file()
        #self._print_best()
        #self._print_best_to_file()

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
        return device, list_ids
    

    def _save_checkpoint(self, epoch, save_best=True):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint_full_no_gumbel.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model_full_no_gumbel.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

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


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        #self.train_dataloader = [(images_id, images.to("cuda:0"), reports_ids.to("cuda:0"), reports_masks.to("cuda:0")) for images_id,images, reports_ids, reports_masks in self.train_dataloader]

        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        print('entering train_epoch')
        train_loss = 0
        # Print the device indices, names, and identifiers
        print('printing device id:', self.device)
        # Assuming `model` is your PyTorch model
        #check_tensor_device(self.model)
        count=0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, seq_length) in tqdm(enumerate(self.train_dataloader)):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            count+=1
        llm_loss= train_loss / len(self.train_dataloader)
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        count_val=0
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, seq_length) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                latent,seq, gs_logps = self.model(images, mode='sample')
                # Separate 'findings' and 'impression' based on [SEP] token
                reports = self.model.tokenizer.decode_batch(seq.cpu().numpy())
                #print(reports)
                
                #parts = reports[0].split(sep_index, 1)  # Split into two parts, assuming the first [SEP] separates 'findings' and 'impression'
                #findings = parts[0].strip() if len(parts) > 0 else ''
                #impression = parts[1].strip() if len(parts) > 1 else ''
                #print('reports impression: ', impression)
                #reports = self.model.tokenizer.decode_batch(seq.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met, val_met_ind = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            #print('ground truths: ', val_gts)
            #print('------------')
            #print('------------')
            print('prediction: ', val_res)
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        '''self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output,_ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        '''
        self.lr_scheduler.step()

        return log, llm_loss
