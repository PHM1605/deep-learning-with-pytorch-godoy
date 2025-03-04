import numpy as np
import datetime 
import torch 
import torch.nn as nn
import random 
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 

class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model 
        self.loss_fn = loss_fn
        self.optimizer = optimizer 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_loader = None 
        self.val_loader = None 
        self.writer = None 
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0 
        self.visualization = {} # to store features after each layer
        self.handles = {} # to remove hooks from each layer after using
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()
    
    def to(self, device):
        try:
            self.device = device 
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead")
            self.model.to(self.device)
            
