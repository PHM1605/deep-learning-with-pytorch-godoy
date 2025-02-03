import numpy as np
from sklearn.linear_model import LinearRegression 
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader 
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter 
import matplotlib.pyplot as plt

%run -i data_generation/simple_linear_regression.py
%run -i data_preparation/v0.py 
%run -i model_configuration/v0.py

n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train_tensor)
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print("Model state dict: ", model.state_dict())

# high-order function for training
def make_train_step_fn(model, loss_fn, optimizer):
    def perform_train_step_fn(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return perform_train_step_fn

%run -i data_preparation/v0.py
%run -i model_configuration/v1.py 
%run -i model_training/v1.py
print("Model after training state dict: ", model.state_dict())

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor 
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    def __len__(self):
        return len(self.x)
# still tensor in CPU
x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()
train_data = CustomDataset(x_train_tensor, y_train_tensor)
print("Example training sample with CustomDataset: ", train_data[0])

# Use built-in TensorDataset
train_data = TensorDataset(x_train_tensor, y_train_tensor)
print("Example training sample with TensorDataset: ", train_data[0])

# Try DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
print("DataLoader trial: ", next(iter(train_loader)))
%run -i data_preparation/v1.py
%run -i model_configuration/v1.py # to get train_step_fn
%run -i model_training/v2.py 

 