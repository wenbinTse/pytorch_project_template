from torch.optim import lr_scheduler
from torch import optim

from options import args

class Scheduler:
    def __init__(self, optimizer):
        if args.lr_scheduler == 'cos':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 32)
        elif args.lr_scheduler == 'exp':
            scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        else:
            raise NotImplementedError()
        self.scheduler = scheduler
    
    def step(self):
        current_epoch = self.scheduler.last_epoch + 1
        if args.lr_scheduler == 'exp' and current_epoch % 5 == 0:
            self.scheduler.step()
        else:
            self.scheduler.step()
