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
#fig = figure3(x_train, y_train)
