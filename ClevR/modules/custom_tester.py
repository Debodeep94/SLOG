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
from pycocoevalcap.bleu.bleu import Bleu
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch

class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        self.ann_path = args.ann_path
        self._load_checkpoint(args.load)

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
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            images_ids, latent_rep, test_gts, test_res, weights = [], [], [], [], []
            count=0
            for batch_idx, (images_id, images, reports_ids, reports_masks) in (enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                print('entering inference...')
                print('image id: ', images_id)
                output, _ = self.model(images, mode='sample')
                #print(_)
                #print(_.size())
                #print(f'tokens from tester:{output.cpu().numpy()}')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                #print('report printing')
                #print('reports: ', reports)
                #attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                    #self.model.encoder_decoder.model.decoder.layers]
                #print('weights', attention_weights)
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                #print('ground_truths', ground_truths)
                bleu_score = self.metric_ftns({i: [gt] for i, gt in enumerate(ground_truths)},
                                        {i: [re] for i, re in enumerate(reports)})
                #print('bleu score', bleu_score)
                wts=bleu_score['BLEU_4']
                weights.append(wts)
                #print('mean_weights ',mean_weights)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                latent_rep.extend(_)
                images_ids.extend(images_id)
                count+=1
                #print('latent_rep', latent_rep)
                with open("file.txt", "w") as output_text:
                    output_text.write(str(test_res))
                if count == 200: # this needs to be user defined
                    break
            #print('predicted', test_res)
            #print('ground truth', test_gts)
            print('entering printing log')
            #print(images_ids)
            #print(len(latent_rep))
            normed_weights = [float(i)/sum(weights) for i in weights]
            tensor_dict = dict(zip(images_ids, latent_rep))
            print('length of dict: ', len(tensor_dict))
            #print('tensor dict: ',tensor_dict)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)
            
        
        # creating data for surrogate model
        self.logger.info('Generating the csv with the reports...')
        chex=pd.read_csv(r"/nfs/data_chaos/dbanerjee/my_data/R2Gen/data/mimic/chexpert_converted.csv")
        chex=chex[['study_id', 'norm_info_score']]
        print(chex.head())

        # Converting chexp to dictionary
        chex_dict= chex.set_index('study_id').T.to_dict('list')
        
        # Extract the common keys between chex and tensor dict
        score_bucket=[]
        for key1 in set(tensor_dict):
            for key2 in set(chex_dict):
                if key1 == key2:
                    print('Common key found. Storing the info score..')
                    info_sc = chex_dict[key2]
                    score_bucket.append(info_sc)

        print('length of score bucket is: ', len(score_bucket))
        tensors= list(tensor_dict.values())
        #print(tensors)

        #train_x=tensors[:int(len(tensors)*0.8)]
        #train_y=np.array(score_bucket[:int(len(score_bucket)*0.8)])
        test_x=tensors
        test_y=np.array(score_bucket)
        #print('train_x', train_x)
        #print(len(train_x))
        #print('train_y', train_y)

        
        test_list=[]
        for i in test_x:
            #i=i.cpu()
            new_list=i.tolist()
            test_list.append(new_list)
        test_list=np.array(test_list)

        
        #print(train_list)
        #minmaxscaler
        scaler_train = MinMaxScaler()
        scaler_test = MinMaxScaler()
        # transform data
        test_x = scaler_test.fit_transform(test_list.reshape(len(test_list),512))
        test_y = scaler_test.fit_transform(test_y.reshape(len(test_y),1))

        # Assume you have independent variables X and a dependent variable 

        test_x_pt = torch.tensor(test_x,dtype=torch.float)
        test_y_pt = torch.tensor(test_y, dtype=torch.float).reshape(-1,1)
        # Instantiate the model
        input_dim = test_x_pt.size(1)
        output_dim = test_y_pt.size(1)
        model = torch.load('/nfs/data_chaos/dbanerjee/my_data/R2Gen/surrogate/surr_model.pt')
        # Define the loss function and optimizer
        criterion1 = torch.nn.MSELoss()
        criterion2 = SurrogateLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Test the model
        with torch.no_grad():
            predicted = model(test_x_pt)
            #print('Predicted values: ', predicted)

        y_preds=np.asarray(scaler_test.inverse_transform(predicted))
        print(y_preds)
        gt=torch.tensor(scaler_test.inverse_transform(test_y_pt))
        mse= criterion1(torch.tensor(y_preds), gt)
        rmse=mse**0.5
        print('the rmse loss is ', rmse)
        print('variance of test list: ', np.var(np.array(score_bucket)))
        return log
    
    '''def crate_csv(self):
        self.logger.info('Generating the csv with the reports...')
        self.ann_path = args.ann_path
        data=self.ann_path#json.load(f)
        report_csv=pd.DataFrame(data.get('test')) # 
        text_file=open('file.txt', 'r')
        pred_reports=text_file.read()
        pred_reports=ast.literal_eval(pred_reports)
        report_csv['predicted_reports']=pred_reports
        report_csv.to_csv('pred_reports_latent.csv')


    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     self.model.encoder_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(report)
                    for word_idx, (attn, word) in enumerate(zip(attns, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn)
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)
    '''