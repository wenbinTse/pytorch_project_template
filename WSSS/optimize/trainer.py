import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from .loss import ComposeLoss
import utils
from utils import info
from options import args

class Trainer:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, 
        test_dataloader: DataLoader, criterion: ComposeLoss, optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.best_score = -1e10
        self.best_epoch = -1
    
    def train(self, epoch):
        # epoch starts from 0
        self.criterion.clear()
        tbar = tqdm(self.train_dataloader)
        recoder = utils.recorder.AccuracyRecorder()
        for i, (image, class_id, file_name) in enumerate(tbar):
            self.optimizer.zero_grad()
            image = image.cuda()
            class_id = class_id.cuda()
            meta = utils.init_meta()
            y, meta = self.model(image, meta)
            loss = self.criterion(logit=y, label=class_id, meta=meta)
            loss.backward()
            self.optimizer.step()
            recoder.add(y, class_id)
        
        self.criterion.print_log()
        self.lr_scheduler.step()

        score = recoder.get_score()
        info(f'Accuracy: {score}')
        utils.dist_barrier()
    
    def test(self, epoch):
        # epoch starts from 0
        with torch.no_grad():
            self.criterion.clear()
            tbar = tqdm(self.test_dataloader)
            recoder = utils.recorder.AccuracyRecorder()
            for i, (image, class_id, file_name) in enumerate(tbar):
                image = image.cuda()
                class_id = class_id.cuda()
                meta = utils.init_meta()
                y, meta = self.model(image, meta)
                self.criterion(logit=y, label=class_id, meta=meta)
                recoder.add(y, class_id)
            
            self.criterion.print_log()

            score = recoder.get_score()
            if score > self.best_score:
                self.best_score = score
                self.best_epoch = epoch
            self.save_model(os.path.join(args.save_dir, 'best_model.pt'))
            info(f'Accuracy: {score}[Epoch{epoch}], Best Accrucay: {self.best_score}[Epoch{self.best_epoch}]')

            if epoch == args.epochs - 1:
                self.save_model(os.path.join(args.save_dir, 'latest_model.pt'))
    
    def save_model(self, path):
        info(f'Saving to {path}')
        if utils.is_primary_device():
            torch.save(self.model.state_dict(), path)
        utils.dist_barrier()
