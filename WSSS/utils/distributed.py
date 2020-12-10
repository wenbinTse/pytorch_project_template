from torch import distributed as dist
import torch
import numpy as np
import random
import os

from options import args
from utils import info

def init_distributed_env():
    torch.backends.cudnn.benchmark = True
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device_count = torch.cuda.device_count()
    if device_count > 1:
        assert os.getenv('RANK') is not None, \
            'When using multiple GPUs, please use torch.distributed.launch to init the environment '
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
    
    info('Using {} GPUs'.format(device_count))

def init_distributed_model(model: torch.nn.Module) -> torch.nn.Module :
    model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    return model

def is_primary_device():
    return not (dist.is_initialized() and dist.get_rank() != 0)

def dist_barrier():
    if dist.is_initialized():
        dist.barrier()