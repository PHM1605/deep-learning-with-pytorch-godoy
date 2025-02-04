import numpy as np
from sklearn.linear_model import LinearRegression 
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader 
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter 
import matplotlib.pyplot as plt
from plots.chapter2 import plot_losses, plot_resumed_losses

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
print("Data Loader training model state dict: ", model.state_dict())

# Organize into function of mini-batch
def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    loss = np.mean(mini_batch_losses)
    return loss 
%run -i model_training/v3.py
print("Minibatch training model state dict: ", model.state_dict())

# train_test_split
%run -i data_preparation/v2.py

# Evaluation
def make_val_step_fn(model, loss_fn):
    def perform_val_step_fn(x, y):
        model.eval()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        return loss.item()
    return perform_val_step_fn 
%run -i model_configuration/v2.py
%run -i model_training/v4.py
print("Model state dict after evaluation: ", model.state_dict())
fig = plot_losses(losses, val_losses)

# Tensorboard
%load_ext tensorboard 
%tensorboard --logdir runs
writer = SummaryWriter('runs/test')
sample_x, sample_y = next(iter(train_loader))
writer.add_graph(model, sample_x.to(device))
writer.add_scalars('loss', {'training': loss, 'validation': val_loss}, epoch)
%run -i data_preparation/v2.py
%run -i model_configuration/v3.py 
%run -i model_training/v5.py
print("Model after training and logging tensorboard: ", model.state_dict())

# Saving model
checkpoint = {
    'epoch': n_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': losses,
    'val_loss': val_losses
    }
torch.save(checkpoint, 'model_checkpoint.pth')
# Loading model back
%run -i data_preparation/v2.py
%run -i model_configuration/v3.py
print("State of untrained model: ", model.state_dict())
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
saved_epoch = checkpoint['epoch']
saved_losses = checkpoint['loss']
saved_val_losses = checkpoint['val_loss']
model.train() # always use TRAIN for resuming training
print("State of trained model at 200 epochs: ", model.state_dict())
%run -i model_training/v5.py # training for another 200 epochs -> 400 epochs totally
print("State of trained model at 400 epochs:  ", model.state_dict())
fig = plot_resumed_losses(saved_epoch, saved_losses, saved_val_losses, n_epochs, losses, val_losses)

# Loading & making predictions
%run -i model_configuration/v3.py
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print("Model to do predict: ", model.state_dict())
new_inputs = torch.tensor([[0.20], [0.34], [0.57]])
model.eval()
print("Predictions: ", model(new_inputs.to(device)))

# # Code snippet to clean weird plots on Tensorboard
# import shutil
# shutil.rmtree('./runs/simple_linear_regression/', ignore_errors=True)