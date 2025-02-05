import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F 
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.datasets import make_moons 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc 
from stepbystep.v0 import StepByStep 
from plots.chapter3 import figure1, figure2, figure3, figure4

X, y = make_moons(n_samples=100, noise=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_val = sc.transform(X_val)
fig = figure1(X_train, y_train, X_val, y_val)

# Data loaders preparation
torch.manual_seed(13)
x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train.reshape(-1,1)).float()
x_val_tensor = torch.as_tensor(X_val).float()
y_val_tensor = torch.as_tensor(y_val.reshape(-1,1)).float() 
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)

# Odds Ratio
def odds_ratio(prob):
    return prob/(1-prob)
p = 0.75
q = 1 - p
fig = figure2(p)

# Log Odds Ratio
def log_odds_ratio(prob):
    return np.log(odds_ratio(prob))
p = 0.75
q = 1 - p
print("Log odds ratio: ", log_odds_ratio(p), log_odds_ratio(q))
fig = figure3(p)

# Sigmoid - inverse function of LogOddsRatio
# i.e. z=LogOddsRatio(p) == p=Sigmoid(z); z is "logit" = weight*x
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
p = 0.75
q = 1 - p
print("Sigmoid of log odds ratio is probability: ", sigmoid(log_odds_ratio(p)), sigmoid(log_odds_ratio(q)))
fig = figure4(p)

# Build logistic regression
torch.manual_seed(42)
model1 = nn.Sequential()
model1.add_module('linear', nn.Linear(2, 1))
model1.add_module('sigmoid', nn.Sigmoid())
print("Model state dict: ", model1.state_dict())

# Dummy data points
dummy_labels = torch.tensor([1.0, 0.0])
