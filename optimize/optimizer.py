from torch import optim
import torch.nn as nn

from options import args

def get_optimizer(model: nn.Module):
    trainable_params = [x for x in model.parameters() if x.require_grad]
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            trainable_params,
            args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimzer == 'sgd':
        optimizer = optim.SGD(
            trainable_params,
            args.lr,
            momentum=args.momentum
        )
    else:
        raise NotImplementedError()
    
    return optimizer
