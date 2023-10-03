import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
        if args.surrogate_model is not None:
            self.surrogate_model = args.surrogate_model

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        #print('target',targets)
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            
            #print(f'lat_rep:{latent_representation}')
            #print(output[-1])
        else:
            raise ValueError
        #print('output',output)
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            latent, output = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output, latent

        elif mode == 'surrogate':
            logits, output = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            loaded_data = torch.load('/home/debodeep.banerjee/R2Gen/train_min_max_scalar_only_find.pt')
            #print('test_x_ft after minmax: ', logits)
            # Retrieve the min and max tensors
            min_tensor = loaded_data['min']
            max_tensor = loaded_data['max']

            min_tensor = min_tensor.to(torch.float32)
            min_tensor = min_tensor.to(torch.float32)

            min_tensor = min_tensor.to(logits.device)
            max_tensor = max_tensor.to(logits.device)
            #print('min_tensor: ', min_tensor)
            #print('max_tensor: ', max_tensor)

            test_x_ft = logits.sub(min_tensor).div(max_tensor - min_tensor)
            test_x_ft = test_x_ft.to(torch.float32)
            #print('test_x_ft after minmax: ', test_x_ft)
            #test_x_tr = latent# torch_minmax(latent, self.device)
            #print(self.surrogate_model)
            surrogate = torch.load(self.surrogate_model)   
            surrogate = surrogate.to(logits.device)
            predicted_ft = surrogate(test_x_ft)
            return output, logits, predicted_ft
        else:
            raise ValueError
        

