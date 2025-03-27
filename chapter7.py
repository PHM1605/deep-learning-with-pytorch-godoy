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

## Auxiliari classifier
from torchvision.models.inception import Inception_V3_Weights

model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
freeze_model(model)
torch.manual_seed(42)
model.AuxLogits.fc = nn.Linear(768, 3)
model.fc = nn.Linear(2048, 3)

# Note: we cannot use the pre-extracted feature approach 
def inception_loss(outputs, labels):
    try: 
        main, aux = outputs
    except ValueError:
        main = outputs 
        aux = None 
        loss_aux = 0
    multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')    
    loss_main = multi_loss_fn(main, labels)
    if aux is not None:
        loss_aux = multi_loss_fn(aux, labels)
    return loss_main + 0.4 * loss_aux 

optimizer_model = optim.Adam(model.parameters(), lr=3e-4)
sbs_incep = StepByStep(model, inception_loss, optimizer_model)
normalizer = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
composer = Compose([
    Resize(299),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    normalizer
])
train_data = ImageFolder(root='rps', transform=composer)
val_data = ImageFolder(root='rps-test-set', transform=composer)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
sbs_incep.set_loaders(train_loader, val_loader)
sbs_incep.train(1)
# Evaluate
print("Evaluation recall of Inception net after 1 epoch: ", StepByStep.loader_apply(val_loader, sbs_incep.correct))

## Each filter in 1x1 convolution is a weighted average of the input channels
# Example: using 1x1 convolution to convert RGB- to grayscale image
scissors = Image.open('rps/scissors/scissors01-001.png')
image = ToDtype(torch.float32, scale=True)(ToImage()(scissors))[:3,:,:].view(1,3,300,300)
weights = torch.tensor([0.2126, 0.7152, 0.0722]).view(1,3,1,1)
convolved = F.conv2d(input=image, weight=weights)
converted = ToPILImage()(convolved[0])
grayscale = scissors.convert('L')
fig = compare_grayscale(converted, grayscale)

## Inception module: StandardType (only one 1x1 Conv) and DimensionReductionType (each ConvolutionBranch has one 1x1 Conv) 
class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        # 1st branch 
        self.branch1x1_1 = nn.Conv2d(in_channels, 2, kernel_size=1) # [2,H,W]
        # 2nd branch
        self.branch5x5_1 = nn.Conv2d(in_channels, 2, kernel_size=1) # [2,H,W]
        self.branch5x5_2 = nn.Conv2d(2, 3, kernel_size=5, padding=2) # [3,H,W]
        # 3rd branch
        self.branch3x3_1 = nn.Conv2d(in_channels, 2, kernel_size=1) # [2,H,W]
        self.branch3x3_2 = nn.Conv2d(2, 3, kernel_size=3, padding=1) # [3,H,W]
        # 4th branch 
        self.branch_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1) # [in_channels,H,W]
        self.branch_pool_2 = nn.Conv2d(in_channels, 2, kernel_size=1) # [2,H,W]
        
    def forward(self, x):
        # 1st branch 
        branch1x1 = self.branch1x1_1(x)
        # 2nd branch
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        # 3rd branch
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        # 4th branch
        branch_pool = self.branch_pool_1(x)
        branch_pool = self.branch_pool_2(branch_pool)
        # Output
        outputs = torch.cat([branch1x1, branch5x5, branch3x3, branch_pool], 1) # [2+3+3+2,H,W]=[10,H,W]
        return outputs 

## Test pushing 3x300x300 image through Inception module
inception = Inception(in_channels=3)
output = inception(image)
print("Output shape through Inception module: ", output.shape) # [1,10,300,300]

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

## Batch Normalization = Normalization + AffineTransformation to mitigate Internal Covariate Shift (ICS)
# Notice: we shouldn't use 'bias' for the layer before BatchNorm, as it will be cancelled out regardless
torch.manual_seed(23)
# randn: random but standard normalized; rand: random but uniform on range [0,1)
dummy_points = torch.randn((200,2)) + torch.rand((200,2))*2 
dummy_labels = torch.randint(2, (200,1))
dummy_dataset = TensorDataset(dummy_points, dummy_labels)
dummy_loader = DataLoader(dummy_dataset, batch_size=64, shuffle=True)
iterator = iter(dummy_loader)
batch1 = next(iterator)
batch2 = next(iterator)
batch3 = next(iterator)
mean1, var1 = batch1[0].mean(axis=0), batch1[0].var(axis=0)
print("Mean and var of the 1st dummy batch: ", mean1, var1)
fig = before_batchnorm(batch1)
