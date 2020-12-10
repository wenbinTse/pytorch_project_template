import torch
from torch.utils.data import DataLoader
from torch import distributed as dist

from .voc_dataset import Classification as VOCClassification
from options import args

def get_dataloader(mode: str):
    assert mode in ['train', 'test']
    if args.dataset == 'voc':
        dataset = VOCClassification(args, mode)
    else:
        raise NotImplementedError()
    sampler = torch.utils.data.distributed.DistributedSampler() if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader
