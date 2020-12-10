import numpy as np
from sklearn.metrics import accuracy_score
import torch

class AccuracyRecorder:
    def __init__(self):
        self.predict = np.array([])
        self.gt = np.array([])
    
    def add(self, logit: torch.Tensor, label: torch.Tensor):
        logit = torch.argmax(logit, dim=-1).flatten().detach().cpu().numpy()
        label = label.flatten().detach().cpu().numpy()
        self.predict = np.concatenate((self.predict, logit))
        self.gt = np.concatenate((self.gt, label))
    
    def get_score(self):
        return accuracy_score(self.gt, self.predict)
