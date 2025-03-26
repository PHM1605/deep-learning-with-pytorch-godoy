import numpy as np
from PIL import Image 
import torch
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision.transforms.v2 import Compose, ToImage, Normalize, Resize, ToPILImage, CenterCrop, RandomResizedCrop, ToDtype 
from torchvision.datasets import ImageFolder 
from torchvision.models import alexnet, resnet18, inception_v3 
from torchvision.models.alexnet import AlexNet_Weights
from torch.hub import load_state_dict_from_url 
from stepbystep.v3 import StepByStep 
from plots.chapter7 import *
from data_generation.rps import download_rps

## Comparing Architectures
fig = figure1()
alex = alexnet(weights=None)
print("AlexNet structure: ", alex)

# ## Adaptive Average Pooling illustration
# result1 = F.adaptive_avg_pool2d(torch.randn(16,32,32), output_size=(6,6))
# result2 = F.adaptive_avg_pool2d(torch.randn(16,12,12), output_size=(6,6))
# print("Adaptive pooling output shapes: ", result1.shape, result2.shape)

## Loading weights from url
weights = AlexNet_Weights.DEFAULT
url = weights.url
print("URL of weights: ", url)
state_dict = load_state_dict_from_url(
    url, model_dir='pretrained', progress=True
)
alex.load_state_dict(state_dict)

def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False 
freeze_model(alex)
print("Alex classifier part: ", alex.classifier)
# Replace last part of the classifier to have only 3 classes (rock/paper/scissor)
alex.classifier[6] = nn.Linear(4096, 3) # =>this part is not frozen
print("Unfrozen part: ")
for name, param in alex.named_parameters():
    if param.requires_grad == True:
        print(name)

## Preparing model
torch.manual_seed(17)
multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer_alex = optim.Adam(alex.parameters(), lr=3e-4)
sbs_alex = StepByStep(alex, multi_loss_fn, optimizer_alex)

## Prepare data
download_rps()
# these values are from the ILSVRC dataset
normalizer = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
composer = Compose([
    Resize(256),
    CenterCrop(224),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    normalizer
])
train_data = ImageFolder(root='rps', transform=composer)
val_data = ImageFolder(root='rps-test-set', transform=composer)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# ## Model training
# sbs_alex.set_loaders(train_loader, val_loader)
# # Notice that it takes too long even though training only 1 epoch and most of the model is frozen 
# sbs_alex.train(1) 
# # ...but it still works very good (96.51% recall for validation set)
# print("Classification recall after 1 epoch only: ", StepByStep.loader_apply(val_loader, sbs_alex.correct))

# ## Solution for above problem: Generating a Dataset of features (Notice: no data-augmentation possible)
# # Step 1: keep only the frozen layer => replacing trainable layer(s) with Identity layer
# alex.classifier[6] = nn.Identity()
# print("New classifier of AlexNet: ", alex.classifier)
# # Step 2: run the whole dataset through it and collect its output as a "dataset of features"
# def preprocessed_dataset(model, loader, device=None):
#     if device is None:
#         device = next(model.parameters()).device 
#     features = None 
#     labels = None 
#     for i, (x,y) in enumerate(loader):
#         model.eval()
#         output = model(x.to(device))
#         if i==0:
#             features = output.detach().cpu()
#             labels = y.cpu()
#         else:
#             features = torch.cat([features, output.detach().cpu()])
#             labels = torch.cat([labels, y.cpu()])
#     dataset = TensorDataset(features, labels)
#     return dataset 
# train_preproc = preprocessed_dataset(alex, train_loader)
# val_preproc = preprocessed_dataset(alex, val_loader)
# torch.save(train_preproc.tensors, 'rps_preproc.pth')
# torch.save(val_preproc.tensors, 'rps_val_preproc.pth')

# Load saved features
x, y = torch.load('rps_preproc.pth')
train_preproc = TensorDataset(x, y)
val_preproc = TensorDataset(*torch.load('rps_val_preproc.pth'))
train_preproc_loader = DataLoader(
    train_preproc, batch_size=16, shuffle=True
)
val_preproc_loader = DataLoader(
    val_preproc, batch_size=16
)

# Step 3: Train the classifier model at the top separately, using feature data
torch.manual_seed(17)
top_model = nn.Sequential(nn.Linear(4096, 3))
multi_loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
optimizer_top = optim.Adam(top_model.parameters(), lr=3e-4)
sbs_top = StepByStep(top_model, multi_loss_fn, optimizer_top)
sbs_top.set_loaders(train_preproc_loader, val_preproc_loader)
sbs_top.train(10) # notice: really fast!!!

# Step 4: Attach the trained top-model to the top of frozen-model
sbs_alex.model.classifier[6] = top_model 
print("Transferred-learning AlexNet model: ", sbs_alex.model.classifier)

## Test this new model on the whole Validation Dataset
print("Transferred-learning AlexNet model validation result: ", StepByStep.loader_apply(val_loader, sbs_alex.correct)) # recall: 96.51%
