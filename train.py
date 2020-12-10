import torch
from torch import distributed as dist

from options import args
from data import get_dataloader
from utils import info, init_distributed

init_distributed()

for key, val in args.__dict__.items():
    info('{:15}: {}'.format(key, val))
