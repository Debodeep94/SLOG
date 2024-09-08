from surrogate_module.surrogate_utils import *
from surrogate_module.rnn_surrogate import *
from abc import abstractmethod
from tqdm import tqdm
import numpy as np
import logging
import torch
# custom modules
from .annotator_loc import *
from .utils import ce_metric,calc_ce
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
        #self.info_score_data= args.info_score_data
        self.surrogate_model = args.surrogate_model
        #self.min_max_scaler = args.min_max_scaler
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
            all_infer = []
            all_gt = []
            count=0
            for batch_idx, (images_id, images, reports_ids, reports_masks,seq_length) in tqdm(enumerate(self.surrogate_dataloader)):
                images, reports_ids, reports_masks, seq_length = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), seq_length.to(self.device)
                # print('entering inference...')
                #print('image id: ', images_id)
                _, seq, seq_logps = self.model(images, mode='sample')
                #print('logps: ', gs_logps)
                # print('logps size: ', seq_logps.size())
                #latent= torch.split(sample_lps, split_size_or_sections=1, dim=0)
                reports = self.model.tokenizer.decode_batch(seq.cpu().numpy())
                # print('report printing')
                sequence_lengths = torch.tensor([len(k) for k in seq]).to(torch.int64)
                #print('pred seq: ', seq)
                #print('seq length: ', sequence_lengths)
                embeddings= self.model.encoder_decoder.model.tgt_embed
                # we don't need this if we  are not using the embeddings
                img_ids.extend(images_id)

                # Surrogate evaluation
                # apply embeddings on both ground truth and prediction. apply surrogate on both. check the accuracy
                embedded_sentence_test = embeddings(seq)
                surrogate_checkpoint = torch.load(self.surrogate_model) 
                surrogate = VisualSurrogate(2048, 512, 4, 256, 6, 5, 2, mode='CELoss')
                # Create a new state dict without the `module.` prefix
                new_state_dict = {}
                for k, v in surrogate_checkpoint.items():
                    new_key = k.replace('module.', '')  # Remove 'module.' prefix
                    new_state_dict[new_key] = v

                # Load the new state dict into the model
                surrogate.load_state_dict(new_state_dict)
                surrogate = surrogate.to(self.device)
                pred_on_infer = surrogate(images, embedded_sentence_test)
                pred_on_infer = pred_on_infer[:,:,1]
                all_infer.extend(pred_on_infer)
                # pred_on_gt = surrogate(images, embedded_sentence_gt)
                # pred_on_gt = torch.sigmoid(pred_on_gt).round()
                # all_gt.append(pred_on_gt.cpu())
                
                reports = self.model.tokenizer.decode_batch(seq.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                
                # latent_rep_check.extend(latent)
                # pred_info_scores.extend(predicted)
                count = count + 1 
                #if count == 5:
                    #break
        # all_infer = np.concatenate(all_infer, axis=0)
        # all_gt = np.concatenate(all_gt, axis=0)
        # torch.save(all_gt,'all_gt_baseline.gt')
        # torch.save(all_infer,'all_infer_baseline.gt')
        test_labels_gt=[]
        for i in test_gts:
            result = check_all_criteria(i)
            test_labels_gt.append(result)  

        test_labels_pred=[]
        for i in test_res:
            result = check_all_criteria(i)
            test_labels_pred.append(result)
        label_set=['cat1', 'cat2', 'cat3', 'cat4', 'cat5']#, 'cat6', 'cat7', 'cat8', 'cat9', 'cat10']
        test_labels_gt=np.array(test_labels_gt)
        test_labels_pred=np.array(test_labels_pred)
        # print(test_labels_gt)
        # print(test_labels_pred)
        # Calculate surrogate performance
        f1_dict=compute_mlc_f1(test_labels_gt,test_labels_pred,label_set)
        prec_dict=compute_mlc_precision(test_labels_gt,test_labels_pred,label_set)
        rec_dict=compute_mlc_recall(test_labels_gt,test_labels_pred,label_set)
        # surrogate_on_preds = compute_mlc_f1(test_labels_gt,all_infer,label_set )
        # surrogate_on_gt = compute_mlc_f1(test_labels_gt,all_gt,label_set )

        test_prec_micro, test_rec_micro=calc_ce(test_gts,test_res)
        test_prec_macro, test_rec_macro=calc_ce(test_gts,test_res, mode='macro')        
        test_met, test_met_id = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
        log.update(**{'test_' + k: v for k, v in test_met.items()})
        log.update(**{'test_' +'LANGUAGE_PRECISION_MICRO': test_prec_micro})
        log.update(**{'test_' +'LANGUAGE_PRECISION_MACRO': test_prec_macro})
        log.update(**{'test_' +'LANGUAGE_RECALL_MICRO': test_rec_micro})
        log.update(**{'test_' +'LANGUAGE_RECALL_MACRO': test_rec_macro})
        concatenated_tensor = torch.cat(all_infer)
        #print('info scores: ', torch.mean(concatenated_tensor,dim=1))
        print(len(pred_info_scores))
        print('mean info scores: ', torch.mean(concatenated_tensor))
        print('median info scores: ', torch.median(concatenated_tensor))
        # print('individual bleu scores: ', test_met_id['BLEU_4'])
        torch.save(test_met_id['BLEU_4'],'base_infer_b4.pt')
        print('test res: ', test_res)
        print('test gts: ', test_gts)  
        print(f'test_ce_metrics_PRECISION: {prec_dict}')
        print(f'test_ce_metrics_RECALL: {rec_dict}')
        print(f'test_ce_metrics_F1: {f1_dict}')
        # print(f'surrogate_on_preds: {surrogate_on_preds}')
        # print(f'surrogate_on_gt: {surrogate_on_gt}')  
        print(log)
        
        return log
