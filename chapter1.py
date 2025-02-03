import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from plots.chapter1 import *

true_b, true_w = 1, 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = 0.1 * np.random.randn(N, 1)
y = true_b + true_w * x + epsilon
idx = np.arange(N)
np.random.shuffle(idx)
train_idx = idx[:int(N * 0.8)]
val_idx = idx[int(N * 0.8):]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
figure1(x_train, y_train, x_val, y_val)

# Random initialization; then train
b = np.random.randn(1)
w = np. random.randn(1)
print("b and w before training: ", b, w)
lr = 0.1
n_epochs = 1000
for epoch in range(n_epochs):
    yhat = b + w * x_train
    error = yhat - y_train # [N,1]
    loss = (error**2).mean() # [1]
    b_grad = 2 * error.mean()
    w_grad = 2 * (x_train*error).mean()
    b = b - lr * b_grad
    w = w - lr * w_grad
print("b and w after training: ", b, w)

# Sanity check with linear regression
linr = LinearRegression()
linr.fit(x_train, y_train)
print("b and w from linear regression: ", linr.intercept_, linr.coef_[0])
fig = figure3(x_train, y_train)

# Test pytorch
scalar = torch.tensor(3.14159)
vector = torch.tensor([1,2,3])
matrix = torch.ones((2,3), dtype=torch.float)
tensor = torch.randn((2,3,4), dtype=torch.float)
print("Torch tensors: ", scalar, vector, matrix, tensor)
print("Tensor size: ", tensor.size(), tensor.shape)
print("Scalar size: ", scalar.size(), scalar.shape)

# view() means reshape, but creates no new tensor (the resulting one is still the old one with different shape))
same_matrix = matrix.view(1, 6)
same_matrix[0, 1] = 2
print("Old matrix:", matrix)
print("Reshaped matrix:", same_matrix)

# clone new matrix
different_matrix = matrix.view(1, 6).clone().detach()
different_matrix[0, 1] = 3 
print("Old matrix: ", matrix)
print("Cloned matrix: ", different_matrix)

# move linear regression from numpy to tensor
x_train_tensor = torch.as_tensor(x_train) # note: if we modify tensor, we are modifying the array too
print("Numpy type: ", x_train.dtype)
print("Torch type: ", x_train_tensor.dtype)
float_tensor = x_train_tensor.float()
print("Convert float64 to float32 result: ", float_tensor.dtype)

# torch.as_tensor(): if we modify the array, we are modifying the tensor too
# torch.tensor() only create a copy
dummy_array = np.array([1, 2, 3])
dummy_tensor = torch.as_tensor(dummy_array)
dummy_tensor_copy = torch.tensor(dummy_array)
dummy_array[1] = 0
print("tensor.as_tensor() when array has been changed: ", dummy_tensor)
print("tensor.tensor() when array has been changed: ", dummy_tensor_copy)

# convert tensor back to numpy 
returned_array = dummy_tensor.numpy()
print("Returned array: ", returned_array)

# check if cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
n_cudas = torch.cuda.device_count()
for i in range(n_cudas):
    print("Device name: ", torch.cuda.get_device_name(i))

# send training data to gpu if available
gpu_tensor = torch.as_tensor(x_train).to(device)
print("gpu_tensor shape:", gpu_tensor.shape)

# convert training data to tensor and send to gpu
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
print("Types of array and tensors: ", type(x_train), type(x_train_tensor), x_train_tensor.type())

# Note: numpy() cannot convert gpu-tensor, must convert to cpu-tensor first
back_to_numpy = x_train_tensor.cpu().numpy()
print("Back to numpy result: ", back_to_numpy)

# For trainable parameters
torch.manual_seed(42)
# this approach is equal to b = torch.randn(...).to(device) then use b.requires_grad_()
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
print("b and w as gpu-tensors and trainable: ", b, w)

# backward() will compute gradient for all "requires_grad" tensors: b, w, yhat, error
yhat = b + w * x_train_tensor
error = yhat - y_train_tensor
loss = (error**2).mean() 
loss.backward()
print("Requires grad? ", error.requires_grad, yhat.requires_grad, b.requires_grad, w.requires_grad)
print("Requires grad? ", y_train_tensor.requires_grad, x_train_tensor.requires_grad)
# check values of the grads
print("Grad values: ", b.grad, w.grad)

# zeroing gradient after update
b.grad.zero_() 
w.grad.zero_()
print("Gradients after zeroing: ", b.grad, w.grad)

# training
n_epochs = 1000
optimizer = optim.SGD([b, w], lr=lr)
loss_fn = nn.MSELoss(reduction='mean')
for epoch in range(n_epochs):
    yhat = b + w * x_train_tensor
    ## Manual loss function
    # error = yhat - y_train_tensor
    # loss = (error**2).mean()
    # loss.backward()
    ## Manual updating and zeroing trainable parameters
    # with torch.no_grad(): # to not losing gradient when re-compute b and w
    #     b -= lr * b.grad
    #     w -= lr * w.grad 
    # b.grad.zero_()
    # w.grad.zero_() 
    ## Loss function with library instead of manual
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()
    ## With optimizer instead of manual
    optimizer.step()
    optimizer.zero_grad()
print("b and w after training: ", b, w)
print("loss: ", loss)
# convert loss to array
print("loss numpy: ", loss.detach().cpu().numpy()) # detach() means removing gradient computation
print("loss numpy (cleaner): ", loss.tolist()) # or loss.item() if one element only
    
# # visualize graph
# b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
# w = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
# yhat = b + w * x_train_tensor 
# error = yhat - y_train_tensor 
# loss = (error**2).mean()
# dot = make_dot(yhat)
# dot.format = 'png'
# dot.render('test.png')

# Linear Regression - manual define Parameters
class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
    def forward(self, x):
        return self.b + self.w * x 
torch.manual_seed(42)
dummy = ManualLinearRegression()
print("Dummy model params: ", list(dummy.parameters()))
print("Dummy model state dict: ", dummy.state_dict())
print("Optimizer state dict: ", optimizer.state_dict())
lr = 0.1
model = ManualLinearRegression().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean')
n_epochs = 1000
for epoch in range(n_epochs):
    model.train() # set model to training mode; turning "dropout" on (otherwise turning off when in "eval" mode)
    yhat = model(x_train_tensor)
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward() # compute gradients for both b and w params
    optimizer.step() # update params using gradients
    optimizer.zero_grad() # zeroing gradient 
print("Model state dict: ", model.state_dict())

# Linear Regression - using nn.Linear
class MyLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)        
    def forward(self, x):
        self.linear(x)

dummy = MyLinearRegression().to(device)
print("nn.Linear Linear Regression params: ", list(dummy.parameters()))
print("nn.Linear Linear Regression state dict: ", dummy.state_dict())
    
# Try using Sequential() - method 1
model = nn.Sequential(nn.Linear(1, 1))
print("Sequential state dict: ", model.state_dict())
# Try using Sequential() - method 2
model = nn.Sequential()
model.add_module('layer1', nn.Linear(1, 1))
model.to(device)
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train_tensor)
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print("Sequential model state dict: ", model.state_dict())