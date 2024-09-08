import torch
import torch.nn as nn
import torchvision.models as models
#import pretrainedmodels
from modules.my_resnet import MyResNet101

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        device=images.device
        self.model=self.model.to(device)
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

    '''
    def forward(self, images):
        patch_feats = self.my_model(images)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.view(batch_size, feat_size, -1).permute(0, 2, 1)
        avg_feats = self.avg_fnt(patch_feats).squeeze()
        return patch_feats, avg_feats
    '''
