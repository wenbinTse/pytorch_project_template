import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from options import args
from utils import info

class CrossEntropyLoss:
    def __call__(self, **kargs):
        return F.cross_entropy(kargs['logit'], kargs['label'])

class MaskLoss:
    def __init__(self):
        pass
    
    def __call__(self, **kargs):
        meta = kargs['meta']
    
class ComposeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        losses = args.loss.split('+')
        self.losses = []
        for i, item in enumerate(losses):
            loss_weight, loss_type = item.split('*')
            loss_weight = float(loss_weight)
            if loss_type == 'ce':
                loss_func = CrossEntropyLoss()
            elif loss_type == 'mask':
                loss_func = MaskLoss()
            else:
                raise NotImplementedError()
            self.losses.append({
                'weight': loss_weight,
                'type': loss_type,
                'func': loss_func
            })

        self.log = np.zeros((0, len(losses) + 1))
    
    def forward(self, **kargs) -> torch.Tensor:
        loss = 0
        new_log = np.zeros((1, len(self.losses) + 1))
        for i, item in enumerate(self.losses):
            t = item['weight'] * item['func'](**kargs)
            new_log[0, i] = t.item()
            loss += t
        new_log[0, -1] = loss.item()
        self.log = np.concatenate([self.log, new_log])
        return loss
    
    def print_log(self):
        for i, item in enumerate(self.losses):
            info('Loss: [{}: {:.4f}]'.format(item['type'], np.mean(self.log[:, i])), end=' ')
        info('[total: {:.4f}]'.format(np.mean(self.log[:, -1])))
    
    def clear(self):
        self.log = np.zeros((0, len(self.losses) + 1))
        




        