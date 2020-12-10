import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

class MaskPredictor(nn.Module):
    def __init__(self, in_channels):
        self.predictor = nn.Sequential({
            nn.Conv2d(in_channels, in_channels // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, 3, stride=1, padding=1),
        })
    
    def forward(self, x: Tensor, meta: dict):
        x = x.detach()
        
        logit = self.predictor(x)
        logit = F.gumbel_softmax(logit, hard=True)
        
        mask = mask[..., 1:]
        meta['masks'].append(mask)

        return mask * x
