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
        if args.min_max_scaler is not None:
            self.min_max_scaler = args.min_max_scaler

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
            latent, seq, seq_logps = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return  latent, seq, seq_logps
        elif mode == 'gumbel':
            latent, seq, gs_logps = self.encoder_decoder(fc_feats, att_feats, mode='diverse_sample')
            return  latent, seq, gs_logps

        elif mode == 'surrogate':
            latent, seq, gs_logps = self.encoder_decoder(fc_feats, att_feats, mode='diverse_sample')
            loaded_data = torch.load(self.min_max_scaler)
            #print('test_x_ft after minmax: ', logits)
            # Retrieve the min and max tensors
            min_tensor = loaded_data['min']
            max_tensor = loaded_data['max']

            min_tensor = min_tensor.to(torch.float32)
            min_tensor = min_tensor.to(torch.float32)

            min_tensor = min_tensor.to(latent.device)
            max_tensor = max_tensor.to(latent.device)
            #print('min_tensor: ', min_tensor)
            #print('max_tensor: ', max_tensor)
            # take the mean of the vectors
            saved_embedding_weights = torch.load('target_embedding_weights.pth')
            new_embedding_layer=nn.Embedding.from_pretrained(saved_embedding_weights)
            #new_embedding_layer.weight.data.copy_(saved_embedding_weights)
            new_embedding_layer=new_embedding_layer.to(seq.device)
            embedded_sentence = new_embedding_layer(seq)
            embedded_sentence = torch.mean(embedded_sentence, dim=1)
            test_x_ft = embedded_sentence.sub(min_tensor).div(max_tensor - min_tensor)
            test_x_ft = test_x_ft.to(torch.float32)
            test_x_ft = test_x_ft/torch.norm(test_x_ft,dim=1, keepdim=True)
            #print('test_x_ft after minmax: ', test_x_ft)
            #test_x_tr = latent# torch_minmax(latent, self.device)
            #print(self.surrogate_model)
            surrogate = torch.load(self.surrogate_model)   
            surrogate = surrogate.to(seq.device)
            predicted_ft = surrogate(test_x_ft)
            return latent, seq, predicted_ft
        else:
            raise ValueError
        

