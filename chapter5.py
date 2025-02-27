import random
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize
from data_generation.image_classification import generate_dataset
from plots.chapter5 import plot_images
from helpers import index_splitter, make_balanced_sampler
from stepbystep.v1 import StepByStep 

## Convolution
# Simple arrays illustration of convolution
single = np.array(
    [[
        [[5,0,8,7,8,1],
         [1,9,5,0,7,7],
         [6,0,2,4,6,6],
         [9,7,6,6,8,4],
         [8,3,8,5,1,3],
         [7,2,7,0,1,0]]
    ]]
)
print("Single shape: ", single.shape) # [1,1,6,6]
identity = np.array([[
    [[0,0,0],
     [0,1,0],
     [0,0,0]]
]])
print("Identity shape: ", identity.shape)
region = single[:, :, 0:3, 0:3]
filtered_region = region * identity
print("First filted region: ", filtered_region)
# Moving around the filter
new_region = single[:, :, 0:3, (0+3):(3+3)]
new_filtered_region = new_region * identity
print("Next filtered region: ", new_filtered_region)

## NOTE: Shape after a convolution: (hi,wi)*(hf,wf) = ((hi+2*padding-hf+1)/stride, (wi+2*padding-wf+1)/stride)

## Convolution with Pytorch
image = torch.as_tensor(single).float()
kernel_identity = torch.as_tensor(identity).float()
convolved = F.conv2d(image, kernel_identity, stride=1)
print("Convolved with torch with functional: ", convolved)
conv = nn.Conv2d(
    in_channels=1, out_channels=1, kernel_size=3, stride=1
)
print("Convolved with torch with module (1 filter): ", conv(image))
conv_multiple = nn.Conv2d(
    in_channels=1, out_channels=2, kernel_size=3, stride=1
)
print("Convolved with torch with module (2 filters): ", conv_multiple(image))
# To set a convolution layer with specific chosen weights
print("TEST: ", conv.weight[0].shape, kernel_identity.shape)
with torch.no_grad():
    conv.weight[0] = kernel_identity
    conv.bias[0] = 0
print("Convolution after setting filter weights to identity: ", conv(image))
# Convolution with stride = 2
convolved_stride2 = F.conv2d(image, kernel_identity, stride=2)
print("Convolution with stride = 2: ", convolved_stride2)

## Padding
constant_padder = nn.ConstantPad2d(padding=1, value=0)
print("Padded with constant image: ", constant_padder(image))
padded = F.pad(image, pad=(1,1,1,1), mode='constant', value=0) # pad in directions [left,right,top,bottom]
print("Padded with functional directions: ", padded)
# Replication padding with F.pad(mode='replicate') or below:
replication_padder = nn.ReplicationPad2d(padding=1)
print("Replication padded image: ", replication_padder(image))
# Reflection padding with F.pad(mode='reflect') or below:
reflection_padder = nn.ReflectionPad2d(padding=1)
print("Reflection padded image: ", reflection_padder(image))
# Circular padding with F.pad(mode='circular') (no module available)
print("Circular padding: ", F.pad(image, pad=(1,1,1,1), mode='circular'))

## Edge filter
edge = np.array([[
    [[0,1,0],
     [1,-4,1],
     [0,1,0]]
    ]])
kernel_edge = torch.as_tensor(edge).float()
print("Edge kernel shape: ", kernel_edge.shape)
padded = F.pad(image, (1,1,1,1), mode='constant', value=0)
conv_padded = F.conv2d(padded, kernel_edge, stride=1)

## Pooling
# Max pool
pooled = F.max_pool2d(conv_padded, kernel_size=2) # can add "stride=1"
maxpool4 = nn.MaxPool2d(kernel_size=4)
pooled4 = maxpool4(conv_padded)
print('Resulting max-pool: ', pooled, pooled4)
# Average pool
# F.avg_pool2d() or nn.AvgPool2d

## Flatten
flattened = nn.Flatten()(pooled)
print("Flattened: ", flattened)
# ... or we can use pooled.view(1, -1)

## LeNet
lenet = nn.Sequential()
# Featurizer
# Block 1: 1x28x28 -> 6x28x28 -> 6x14x14
lenet.add_module('C1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2))
lenet.add_module('func1', nn.ReLU())
lenet.add_module('S2', nn.MaxPool2d(kernel_size=2))
# Block 2: 6x14x14 -> 16x10x10 -> 16x5x5
lenet.add_module('C3', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5))
lenet.add_module('func2', nn.ReLU())
lenet.add_module('S4', nn.MaxPool2d(kernel_size=2))
# Block3: 16x5x5 -> 120x1x1
lenet.add_module('C5', nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5))
lenet.add_module('func2', nn.ReLU())
# Flattening -> 'features' (120,)
lenet.add_module('flatten', nn.Flatten())
# Classification
# Hidden Layer
lenet.add_module('F6', nn.Linear(in_features=120, out_features=84))
lenet.add_module('func3', nn.ReLU())
lenet.add_module('OUTPUT', nn.Linear(in_features=84, out_features=10))

## Multiclass classification problem
images, labels = generate_dataset(img_size=10, n_images=1000, binary=False, seed=17)
fig = plot_images(images, labels, n_plot=30)

class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x 
        self.y = y
        self.transform = transform 
    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        return x, self.y[index]
    def __len__(self):
        return len(self.x)

x_tensor = torch.as_tensor(images/255).float()
y_tensor = torch.as_tensor(labels).long()
train_idx, val_idx = index_splitter(len(x_tensor), [80, 20])
x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]
x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]
train_composer = Compose([Normalize(mean=(0.5,), std=(0.5,))])
val_composer = Compose([Normalize(mean=(0.5,), std=(0.5,))])
train_dataset = TransformedTensorDataset(x_train_tensor, y_train_tensor, transform=train_composer)
val_dataset = TransformedTensorDataset(x_val_tensor, y_val_tensor, transform=val_composer)
sampler = make_balanced_sampler(y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)

## Softmax - the probabilities
logits = torch.tensor([1.3863, 0.0000, -0.6931])
odds_ratio = torch.exp(logits)
print("Odds ratio (exp(z)): ", odds_ratio)
# Other methods: nn.Softmax(dim=-1)(logits); F.softmax(logits, dim=-1)
softmaxed = odds_ratio / odds_ratio.sum()
print("Softmaxed: ", softmaxed)

## Log Softmax: use F.log_softmax() or nn.LogSoftmax
## Negative log likelihood loss: -1/(N0+N1+N2)*(sum-of-N0-predicted-probs-for-class-0+sum-of-N1-predicted-probs-for-class-1+sum-of-N2-predicted-probs-for-class-2)
log_probs = F.log_softmax(logits, dim=-1)
print("Log probs: ", log_probs)
label = torch.tensor([2])
print("Loss when label is 2: ", F.nll_loss(log_probs.view(-1,3), label, reduction='mean')) # reduction: 'mean'/'sum'/'none'; last one returns array of losses

## NLLLoss
torch.manual_seed(11)
dummy_logits = torch.randn((5,3))
dummy_labels = torch.tensor([0,0,1,2,1])
dummy_log_probs = F.log_softmax(dummy_logits, dim=-1)
print("Dummy log probs: ", dummy_log_probs)
loss_fn = nn.NLLLoss()
print("NLLLoss: ", loss_fn(dummy_log_probs, dummy_labels))
loss_fn = nn.NLLLoss(weight=torch.tensor([1.,1.,2.]))
print("NLLLoss with sample weights: ", loss_fn(dummy_log_probs, dummy_labels))
loss_fn = nn.NLLLoss(ignore_index=2)
print("NLLLoss with label y=2 being ignored: ", loss_fn(dummy_log_probs, dummy_labels))

## Cross-Entropy Loss: combine log_softmax  and nllloss into one
torch.manual_seed(11)
dummy_logits = torch.randn((5, 3))
dummy_labels = torch.tensor([0,0,1,2,1])
loss_fn = nn.CrossEntropyLoss()
print("CrossEntropy loss: ", loss_fn(dummy_logits, dummy_labels))


## Model with Cross-Entropy loss
torch.manual_seed(13)
model_cnn1 = nn.Sequential()
# Featurizer
# Block 1: 1@10x10 -> n_channels@8x8 -> n_channels@4x4 -> n_channels*4*4
n_channels = 1
model_cnn1.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3))
model_cnn1.add_module('relu1', nn.ReLU())
model_cnn1.add_module('maxp1', nn.MaxPool2d(kernel_size=2))
model_cnn1.add_module('flatten', nn.Flatten())
# Classification
model_cnn1.add_module('fc1', nn.Linear(in_features=n_channels*4*4, out_features=10))
model_cnn1.add_module('relu2', nn.ReLU())
model_cnn1.add_module('fc2', nn.Linear(in_features=10, out_features=3))
# we could add nn.LogSoftmax layer here, then must use NLLLoss function
# ... or not necessary, then use nn.CrossEntropyLoss function
lr = 0.1
multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer_cnn1 = optim.SGD(model_cnn1.parameters(), lr=lr)
sbs_cnn1 = StepByStep(model_cnn1, multi_loss_fn, optimizer_cnn1)
sbs_cnn1.set_loaders(train_loader, val_loader)
sbs_cnn1.train(20)
fig = sbs_cnn1.plot_losses()
