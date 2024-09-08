import os
from abc import abstractmethod
#from apex import amp
import time
import torch
import torch.nn as nn
import pandas as pd
from numpy import inf
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
#import wandb
from surrogate_module.surrogate_utils import *
from surrogate_module.rnn_surrogate import *
from .annotator_loc import *
# from .annotator import *
from .utils import *
from .ce_metrics import *

print('GPU availability: ', torch.cuda.is_available())
class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        #torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
        print('hhh')
        print(torch.cuda.is_available())
        print(self.device)
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
        if args.load is not None:
            self._load_checkpoint(args.load)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}#,
                              #'test': {self.mnt_metric_test: self.mnt_best}}
        #self.writer = SummaryWriter(log_dir='logs')

        self.surrogate_model = args.surrogate_model
        surrogate_checkpoint = torch.load(self.surrogate_model)   
        self.surrogate = VisualSurrogate(2048,512,8, 512, 4, 5, 2, mode='CELoss')
        self.surrogate = self.surrogate.to(self.device)
        new_state_dict = {}
        for k, v in surrogate_checkpoint.items():
            new_key = k.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[new_key] = v
        self.surrogate.load_state_dict(new_state_dict)
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

    def finetune(self):
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
        print('n_gpu:',n_gpu)
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
            best_path = os.path.join(self.checkpoint_dir, 'finetuned_best_CE_surr_swap_'+str(self.ce_weight).replace('.', '_')+'_sr'+str(self.surr_weight).replace('.', '_')+'.pth')
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
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader):
        super(FineTuner, self).__init__(model, criterion, metric_ftns, optimizer, args)
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
        for batch_idx, (images_id, images, reports_ids, reports_masks, seq_length) in tqdm(enumerate(self.train_dataloader)):
            
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
            
            #images_ft, reports_ids_ft, reports_masks_ft = images_surr_ft.to(self.device), reports_ids_surr_ft.to(self.device), reports_masks_surr_ft.to(self.device)
            print('------------------------------------------------------------------------------------------------')
            print('Entering Training Loop')
            
            
            output = self.model(images, reports_ids, mode='train')

            loss_lang = self.criterion(output, reports_ids, reports_masks)
            print('Cross entropy loss', loss_lang)
            #print(loss_lang.item())
            loss_llm+=loss_lang.item()#.append(loss_lang.item())
            print('cumulative loss: ', loss_llm)
            
            #print('Running sample mode')
            
            #latent, output_surr = self.model(images,reports_ids, mode='sample')
            seq_logps, seq, gs_logps = self.model(images, mode='gumbel')
            # first_elements_per_timestep = gs_logps[:, :, 0]
            # first_elements_per_timestep = torch.exp(gs_logps[:, :, 0])
            # Check shapes for debugging
            print("Shape of seq_logps:", seq_logps.size())  # Should be (batch_size_t, vocab_size)
            print("Shape of gs_logps:", gs_logps.size())
            # print("sequences:", seq[0])
            # print("Shape of gs_logps:", first_elements_per_timestep.size()) 
            # print(f"gs_logps.requires_grad: {first_elements_per_timestep.requires_grad}")  # Should be True
            #lat_vec_ft, seq_ft, gs_logps_ft = self.model(images_ft, mode='sample')
            # for params in self.model.visual_extractor.parameters():
            #     params.requires_grad=False

            # Assuming self.model.encoder_decoder.model.tgt_embed[0].lut gives you the embedding weights
            # embed_weight = self.model.encoder_decoder.model.tgt_embed[0].lut.weight

            # # # Define a linear layer to mimic one-hot embedding lookup
            # onehot_embedding = nn.Linear(embed_weight.shape[0], embed_weight.shape[1], bias=False).to(self.device)

            # # Initialize the linear layer weights with the actual embedding weights
            # # onehot_embedding.weight = nn.Parameter(torch.tensor(embed_weight.transpose())[:, :-2])
            # #             we don't need this if we  are not using the embeddings
            embeddings= self.model.encoder_decoder.model.tgt_embed[0].lut
            embedded_sentence_tr = torch.matmul(torch.exp(gs_logps), embeddings.weight) #commenting for the time being
            # embedded_sentence_tr = onehot_embedding(gs_logps)
            

            # # # we don't need this if we  are not using the embeddings
            # embedded_sentence_tr = embeddings(seq)
            print(f"embedded_sentence_tr.requires_grad: {embedded_sentence_tr.requires_grad}")  # Should be True
            #embedded_sentence_ft = embeddings(seq_ft)
            print(embedded_sentence_tr.size())
            print(self.surrogate_model)
            for params in self.surrogate.parameters():
                params.requires_grad = False
            predicted_tr = self.surrogate(images, embedded_sentence_tr)
            # predicted_tr = predicted_tr[:,:,1]
            surr_tr_score = torch.mean(predicted_tr)
            #print('predicted tr: ', predicted_tr)
            print('predicted tr shape: ', predicted_tr.size())
            print('surr train score: ', surr_tr_score)
            #print('surr fine-tune score: ', surr_ft_score)
            print('ce loss: ', loss_lang)
            #surrogate_loss = surr_tr_score
            tot_surr_loss+=surr_tr_score.item()
            print('surrogate loss: ', surr_tr_score)
            
            surrogate_loss = self.surr_weight*surr_tr_score
            print(f"surrogate_loss.requires_grad: {surrogate_loss.requires_grad}")  # Should be True
            ce_loss = self.ce_weight*loss_lang
            hdm_loss =  ce_loss-surrogate_loss#-0.05*torch.mean(first_elements_per_timestep)
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
            # if count==70:
            #     break  
        log = {'total cross entropy loss: ': loss_llm/ count}
        log.update(**{'total hdm loss: ': hybrid/ count})
        log.update(**{'total surrogate loss: ': tot_surr_loss/ count})

        '''wandb.log({"Cross entropy loss": loss_llm/ count,
                        'surrogate loss': tot_surr_loss/ count,
                        'hdm_loss': hybrid/ count
                        })'''
        print(len(self.train_dataloader))
        print('log: ', log)
        self.model.eval()
        print('entering validation ...')
        #self.model.eval()
        with torch.no_grad():
            count_val=0
            val_gts, val_res, val_inf_pred = [], [],[]
            for batch_idx_val, (images_id, images, reports_ids, reports_masks, seq_length) in tqdm(enumerate(self.val_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                _,seq_val, seq_logps  = self.model(images, mode='sample')
                embeddings= self.model.encoder_decoder.model.tgt_embed[0].lut
                embedded_sentence_val = torch.matmul(torch.exp(seq_logps), embeddings.weight) #commenting for the time being
                predicted_val = self.surrogate(images,embedded_sentence_val)
            
                # predicted_val = predicted_val[:,:,1]
                val_inf_pred.extend(predicted_val)
                reports = self.model.tokenizer.decode_batch(seq_val.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                count_val=count_val+1
                # if count_val==50:
                #     break

            concatenated_tensor = torch.cat(val_inf_pred)
            print('pred text', val_res)
            
            print('grund truth text', val_gts)
            val_met, _ = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                    {i: [re] for i, re in enumerate(val_res)})
            #print(val_met)                       
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        #torch.cuda.empty_cache()
        # print('len val info sc:', len(predicted_val))
        mean_inf_sc = torch.mean(concatenated_tensor)
        #wandb.log({'validation info score':mean_inf_sc})
        count_val_res = 0
        for i in val_res:
            # print(len(i.split()))
            count_val_res += len(i.split())
        avg_len_res = count_val_res/len(val_res)

        count_val_gts = 0
        for i in val_gts:
            # print(len(i.split()))
            count_val_gts += len(i.split())
        avg_len_gts = count_val_gts/len(val_gts)

        log.update(**{'SURROGATE_SCORE: ': mean_inf_sc.item()})
        log.update(**{'AVERAGE_LENGTH_PREDS: ': avg_len_res})
        log.update(**{'AVERAGE_LENGTH_GTS: ': avg_len_gts})
        log.update(**{'SURROGATE_SCORE: ': mean_inf_sc.item()})

        train_labels_gt=[]
        for i in val_gts:
            result = check_all_criteria(i)
            train_labels_gt.append(result)  

        train_labels_pred=[]
        for i in val_res:
            result = check_all_criteria(i)
            train_labels_pred.append(result)
        label_set=['cat1', 'cat2', 'cat3', 'cat4', 'cat5']
        train_labels_gt=np.array(train_labels_gt)
        train_labels_pred=np.array(train_labels_pred)
        val_metrics=compute_mlc_f1(train_labels_gt,train_labels_pred,label_set )
        val_metrics_rec=compute_mlc_recall(train_labels_gt,train_labels_pred,label_set )
        val_metrics_prec=compute_mlc_precision(train_labels_gt,train_labels_pred,label_set )
        val_prec_micro, val_rec_micro=calc_ce(val_gts,val_res)
        val_prec_macro, val_rec_macro=calc_ce(val_gts,val_res, mode='macro')
        print('printing validation metrics...')
        # print(val_metrics)
        pos_f1=val_metrics['F1_MICRO']
        rec_mic=val_metrics_rec['RECALL_MICRO']
        rec_mac=val_metrics_rec['RECALL_MACRO']
        prec_mic=val_metrics_prec['PRECISION_MICRO']
        prec_mac=val_metrics_prec['PRECISION_MACRO']
        log.update(**{'val_' +'POSITIVE_F1': pos_f1})
        log.update(**{'val_' +'RECALL_MACRO': rec_mac})
        log.update(**{'val_' +'RECALL_MICRO': rec_mic})
        log.update(**{'val_' +'PRECISION_MICRO': prec_mic})
        log.update(**{'val_' +'PRECISION_MACRO': prec_mac})
        log.update(**{'val_' +'LANGGUAGE_PRECISION_MICRO': val_prec_micro})
        log.update(**{'val_' +'LANGGUAGE_PRECISION_MACRO': val_prec_macro})
        log.update(**{'val_' +'LANGGUAGE_RECALL_MICRO': val_rec_micro})
        log.update(**{'val_' +'LANGGUAGE_RECALL_MACRO': val_rec_macro})
        self.lr_scheduler.step()
        return log, hdm_loss, val_inf_pred #this should be replaced by hdm_loss
