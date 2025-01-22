import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from surrogate_module.surrogate_utils import *
from surrogate_module.rnn_surrogate import *

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        # Determine forward function based on dataset
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
        
        # Load surrogate model if specified
        if args.surrogate_model is not None:
            self.surrogate_model = args.surrogate_model
        
        for name, param in  self.encoder_decoder.named_parameters():
            if ".decoder" not in name:  # Adjust this based on your decoder layer's naming
                param.requires_grad = False
    
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            _, seq, seq_logps = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return  _, seq, seq_logps
        elif mode == 'finetune':
            _, seq, logps = self.encoder_decoder(fc_feats, att_feats, mode='diverse_sample')
            return  _, seq,logps
        else:
            raise ValueError
