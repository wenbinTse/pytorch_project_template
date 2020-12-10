import torch
from torch import distributed as dist
from tqdm import tqdm

from options import args
from data import get_dataloader
from utils import info, init_distributed_env, init_distributed_model, init_meta

import models
import optimize

init_distributed_env()

for key, val in args.__dict__.items():
    info('{:15}: {}'.format(key, val))

model = models.get_model()
model = init_distributed_model(model)

optimizer = optimize.get_optimizer(model)
scheduler = optimize.Scheduler(optimizer)
criterion = optimize.ComposeLoss()

train_dataloader = get_dataloader('train')
test_dataloader = get_dataloader('test')

trainer = optimize.Trainer(
    model, train_dataloader, test_dataloader, criterion, optimizer, scheduler
)

for epoch in range(args.epochs):
    info(f'Training epoch {epoch}')
    with torch.autograd.set_detect_anomaly(True):
        trainer.train(epoch)
    info(f'Testing epoch {epoch}')
    trainer.test(epoch)

