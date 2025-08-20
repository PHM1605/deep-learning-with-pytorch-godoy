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
from plots.chapter3 import *

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
dummy_predictions = torch.tensor([0.9, 0.2])
# First way
positive_pred = dummy_predictions[dummy_labels==1]
first_summation = torch.log(positive_pred).sum()
negative_pred = dummy_predictions[dummy_labels == 0]
second_summation = torch.log(1 - negative_pred).sum()
n_total = dummy_labels.size(0)
loss = -(first_summation + second_summation) / n_total
print("Loss (first way): ", loss)
# Second way - smarter
summation = torch.sum(dummy_labels*torch.log(dummy_predictions) + (1-dummy_labels)*torch.log(1-dummy_predictions))
loss = -summation / n_total 
print("Loss (smart way): ", loss)

# Binary Cross Entropy With Logits Loss
loss_fn_logits = nn.BCEWithLogitsLoss(reduction='mean')
logit1 = log_odds_ratio(0.9)
logit2 = log_odds_ratio(0.2)
dummy_labels = torch.tensor([1.0, 0.0])
dummy_logits = torch.tensor([logit1, logit2]) # [2.2, -1.4])
loss = loss_fn_logits(dummy_logits, dummy_labels) # 0.16, same as before

# Imbalance dataset - more negative classes
dummy_imb_labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
dummy_imb_logits = torch.tensor([logit1, logit2, logit2, logit2])
n_neg = (dummy_imb_labels==0).sum().float()
n_pos = (dummy_imb_labels==1).sum().float()
pos_weight = (n_neg/n_pos).view(1,) # convert from tensor-scalar to tensor-list-of-1-element
print("Pos weight: ", pos_weight)
# Loss calculated with library
loss_fn_imb = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
loss = loss_fn_imb(dummy_imb_logits, dummy_imb_labels)
print("Weighted average loss: ", loss) # 0.25, bigger
# Loss calculated manually with calculating sum of loss first
loss_fn_imb_sum = nn.BCEWithLogitsLoss(
    reduction = 'sum',
    pos_weight=pos_weight
    )
loss = loss_fn_imb_sum(dummy_imb_logits, dummy_imb_labels)
loss = loss / (pos_weight * n_pos + n_neg) # 0.1643
print("Weighted average loss - true", loss)

# Model configuration
lr = 0.1
torch.manual_seed(42)
model = nn.Sequential()
model.add_module('linear', nn.Linear(2,1))
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.BCEWithLogitsLoss()
# Model Training
n_epochs = 100
sbs = StepByStep(model, loss_fn, optimizer)
sbs.set_loaders(train_loader, val_loader)
sbs.train(n_epochs)
fig = sbs.plot_losses()
print("StepByStep model state dict: ", model.state_dict())
# Prediction
predictions = sbs.predict(x_train_tensor[:4]) # output: logits
probabilities = sigmoid(predictions)
print("Probabilities of predictions: ", probabilities)
classes = (predictions >= 0).astype(int)
print("Classes: ", classes)
# Training set
fig = figure7(X_train, y_train, sbs.model, sbs.device)
# Validation set
fig = figure7(X_val, y_val, sbs.model, sbs.device)

## Are my data points separable?
x = np.array([-2.8, -2.2, -1.8, -1.3, -.4, 0.3, 0.6, 1.3, 1.9, 2.5])
y = np.array([0., 0., 0., 0., 1., 1., 1., 0., 0., 0.])
fig = one_dimension(x, y)
fig = two_dimensions(x, y)
# Expand dimensions with Neural Network
model = nn.Sequential()
model.add_module('hidden', nn.Linear(2, 10))
model.add_module('activation', nn.ReLU())
model.add_module('output', nn.Linear(10, 1))
model.add_module('sigmoid', nn.Sigmoid())
loss_fn = nn.BCELoss()

## Classfication threshold
logits_val = sbs.predict(X_val)
probabilities_val = sigmoid(logits_val).squeeze()
threshold = 0.5
# fig = figure9(X_val, y_val, sbs.model, sbs.device, probabilities_val, threshold)

## Confusion matrix illustration
# fig = figure10(y_val, probabilities_val, threshold, 0.04, False)
fig = figure10(y_val, probabilities_val, threshold, 0.04, True)
cm_thresh50 = confusion_matrix(y_val, probabilities_val>=0.5)
print("Confusion matrix: ", cm_thresh50) # [[TN, FP], [FN, TP]]
def split_cm(cm):
    actual_negative = cm[0]
    tn = actual_negative[0]
    fp = actual_negative[1]
    actual_positive = cm[1]
    fn = actual_positive[0]
    tp = actual_positive[1]
    return tn, fp, fn, tp 
def tpr_fpr(cm):
    tn, fp, fn, tp = split_cm(cm)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr
print("tpr, fpr: ", tpr_fpr(cm_thresh50))

def precision_recall(cm):
    tn, fp, fn, tp = split_cm(cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall 
print("precision, recall: ", precision_recall(cm_thresh50))

## Trade-offs and Curves
fig = eval_curves_from_probs(y_val, probabilities_val, [0.5], annot=True)

print("Confusion matrix for threshold 0.3: ", confusion_matrix(y_val, probabilities_val >= 0.3))
fig = eval_curves_from_probs(y_val, probabilities_val, [0.3, 0.5], annot=True)
print("Confusion matrix for threshold 0.7: ", confusion_matrix(y_val, probabilities_val >= 0.7))
fig = eval_curves_from_probs(y_val, probabilities_val, [0.3, 0.5, 0.7], annot=True)
threshs = np.linspace(0., 1, 11)
fig = figure17(y_val, probabilities_val, threshs)

fpr, tpr, thresholds1 = roc_curve(y_val, probabilities_val)
prec, rec, thresholds2 = precision_recall_curve(y_val, probabilities_val)
fig = eval_curves(fpr, tpr, rec, prec, thresholds1, thresholds2, line=True)

## The precision quirk
fig = figure19(y_val, probabilities_val)

## Best curve
fig = figure20(y_val)

## Worst curve (randomly generated)
np.random.seed(39)
random_probs = np.random.uniform(size=y_val.shape)
fpr_random, tpr_random, thresholds1_random = roc_curve(y_val, random_probs)
prec_random, rec_random, thresholds2_random = precision_recall_curve(y_val, random_probs)
fig = figure21(y_val, random_probs)

## AUC 
auroc = auc(fpr, tpr)
aupr = auc(rec, prec)
print("Our aucs: ", auroc, aupr)
# In theory, worst ROC auc is 0.5; worst precision auc is #positive/#totalsamples i.e. 0.55 in our case
auroc_random = auc(fpr_random, tpr_random)
aupr_random = auc(rec_random, prec_random)
print("Bad aucs: ", auroc_random, aupr_random)

# ## Putting all together
# torch.manual_seed(13)
# x_train_tensor = torch.as_tensor(X_train).float()
# y_train_tensor = torch.as_tensor(y_train.reshape(-1,1)).float()
# x_val_tensor = torch.as_tensor(X_val).float()
# y_val_tensor = torch.as_tensor(y_val.reshape(-1,1)).float() 
# train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
# train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(dataset=val_dataset, batch_size=16)
# lr = 0.1
# torch.manual_seed(42)
# model = nn.Sequential()
# model.add_module('linear', nn.Linear(2,1))
# optimizer = optim.SGD(model.parameters(), lr=lr)
# loss_fn = nn.BCEWithLogitsLoss()
# n_epochs = 100
# sbs = StepByStep(model, loss_fn, optimizer)
# sbs.set_loaders(train_loader, val_loader)
# sbs.train(n_epochs)
# logits_val = sbs.predict(X_val)
# probabilities_val = sigmoid(logits_val).squeeze()
# cm_thresh50 = confusion_matrix(y_val, probabilities_val>=0.5)
# print(cm_thresh50)