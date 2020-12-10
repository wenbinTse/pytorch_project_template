from .resnet import resnet50
from options import args

import torch
import torch.nn as nn

def get_model(num_classes=20) -> nn.Module: 
    if args.model == 'resnet50':
        model = resnet50(pretrained=True, progress=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError()
    return model

__all__ = ['get_model']
