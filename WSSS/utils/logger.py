import logging
from options import args
import os

from torch import distributed as dist

class Logger():
    def __init__(self):
        self.file = open(os.path.join(args.save_dir, 'log.txt'), 'w+')
    
    def info(self, *dargs, **kargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        print(*dargs, **kargs)
        print(*dargs, **kargs, file=self.file)

logger = Logger()
info = logger.info
