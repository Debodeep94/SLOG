import torch
import torch.nn as nn
import torchvision.models as models

class MyResNet101(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        # create features branch using https://github.com/pytorch/vision/blob/2a52c2dca73513d0d0c3e2a505aed05e5b9aa792/torchvision/models/resnet.py#L230-L246
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.to(self.features[0].weight.device)  # Move x to the device of the first convolutional layer
            x = x.to(device) 
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)