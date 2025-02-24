import random
import numpy as np 
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, SubsetRandomSampler 
from torchvision.transforms.v2 import Compose, ToImage, Normalize, ToPILImage, RandomHorizontalFlip, Resize, ToDtype
import matplotlib.pyplot as plt
from data_generation.image_classification import generate_dataset 
from stepbystep.v0 import StepByStep
from plots.chapter4 import *

images, labels = generate_dataset(img_size=5, n_images=300, binary=True, seed=13)
fig = plot_images(images, labels, n_plot=30)

## Images and Channels
image_r = np.zeros((5, 5), dtype=np.uint8)
image_r[:, 0] = 255
image_r[:, 1] = 128
image_g = np.zeros((5, 5), dtype=np.uint8)
image_g[:, 1] = 128
image_g[:, 2] = 255
image_g[:, 3] = 128
image_b = np.zeros((5, 5), dtype=np.uint8)
image_b[:, 3] = 128
image_b[:, 4] = 255
# This formula is to convert RGB image to gray scale image
image_gray = 0.2126 * image_r + 0.7152 * image_g + 0.0722 * image_b 
image_rgb = np.stack([image_r, image_g, image_b], axis=2)
fig = image_channels(image_r, image_g, image_b, image_rgb, image_gray, rows=(0, 1))
fig = image_channels(image_r, image_g, image_b, image_rgb, image_gray, rows=(0, 2))

## Shape: NCHW (Pytorch) vs NHWC (Tensorflow)
print("Images shape: ", images.shape) # [300,1,5,5]
example = images[7] # [1,5,5]
example_hwc = np.transpose(example, (1,2,0)) # [5,5,1]
image_tensor = ToImage()(example_hwc) # convert to tensor, integer and [0,255]
print("Image tensor and shape: ", image_tensor, image_tensor.shape)
print("Is Image a tensor? ", isinstance(image_tensor, torch.Tensor))
example_tensor = ToDtype(torch.float32, scale=True)(image_tensor) # convert from "int [0,255]" to "float [0,1]"
print("Scaled float tensor: ", example_tensor)

## Shorten the above steps
def ToTensor():
    return Compose([ToImage(), ToDtype(torch.float32, scale=True)])
tensorizer = ToTensor()
example_tensor = tensorizer(example_hwc)
print("Scaled float tensor: ", example_tensor)
example_img = ToPILImage()(example_tensor) # convert tensor to PIL Image
print("Type of PIL image: ", type(example_img))
plt.clf()
plt.imshow(example_img, cmap='gray')
plt.grid(False)
plt.savefig('test.png')

## Image transformation with PIL image
flipper = RandomHorizontalFlip(p=1.0)
flipped_img = flipper(example_img)
plt.imshow(flipped_img, cmap='gray')
plt.grid(False)
plt.savefig('test2.png')

## Image transformation with Tensor only
# Standardize for image = MinMaxScaling i.e. all pixels in range [-1,1] => Normalize [0,1]-image with 0.5 mean and 0.5 std
img_tensor = tensorizer(flipped_img) # all pixels in [0,1] range
normalizer = Normalize(mean=(0.5,), std=(0.5,))
normalized_tensor = normalizer(img_tensor)

## Doing a series of transformation with Compose
composer = Compose([
    RandomHorizontalFlip(p=1.0), 
    Normalize(mean=(0.5,), std=(0.5,))
    ])
composed_tensor = composer(example_tensor)
print("Are two methods equal? ", (composed_tensor==normalized_tensor).all())

## Compare between "example" (the original [0,255] CHW integer one, numpy array) and "example_tensor" (the [0,255] CHW float one)
example_tensor = torch.as_tensor(example/255).float()
print("Example tensor: ", example_tensor)

## Data preparation
x_tensor = torch.as_tensor(images/255).float()
y_tensor = torch.as_tensor(labels.reshape(-1, 1)).float() # [300,1]

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
    
composer = Compose([
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=(0.5,), std=(0.5,))
])
dataset = TransformedTensorDataset(x_tensor, y_tensor, composer)
# splits: list of split sizes of train/test
def index_splitter(n, splits, seed=13):
    idx = torch.arange(n)
    splits_tensor = torch.as_tensor(splits)
    total = splits_tensor.sum().float()
    # if total does not add up to 1.0, make the splits adding up to 1.0 e.g. [0.8, 0.2]
    if not total.isclose(torch.ones(1)[0]):
        splits_tensor = splits_tensor/total 
    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)

train_idx, val_idx = index_splitter(len(x_tensor), [80, 20])
print("Train indices: ", train_idx.indices)
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
train_loader = DataLoader(
    dataset=dataset, batch_size=16, sampler=train_sampler 
)
val_loader = DataLoader(
    dataset=dataset, batch_size=16, sampler=val_sampler
)
print(f"#batches in the train dataloader: {len(iter(train_loader))}; #batches in the val dataloader: {len(iter(val_loader))}")

## Doing the train/val split with 2 different composers (no SubsetRandomSampler)
x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]
x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]
train_composer = Compose([
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=(0.5,), std=(0.5,))
])
val_composer = Compose([
    Normalize(mean=(0.5,), std=(0.5,))
])
train_dataset = TransformedTensorDataset(
    x_train_tensor, y_train_tensor, transform=train_composer
)
val_dataset = TransformedTensorDataset(
    x_val_tensor, y_val_tensor, transform=val_composer
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=16, shuffle=True 
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=16
)

## WeightedRandomSampler for imbalance dataset
# List of classes, and count of each class
def make_balanced_sampler(y):
    classes, counts = y.unique(return_counts=True) # tensor([0,1]); tensor([80,160])
    weights = 1.0 / counts.float() # tensor([0.0125, 0.0063])
    sample_weights = weights[y_train_tensor.squeeze().long()] # Tensor: [a,b][0,1,0] = [a,b,a]
    generator = torch.Generator()
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights),
        generator=generator,
        replacement=True
        )
    return sampler 
sampler = make_balanced_sampler(y_train_tensor)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=16, sampler=sampler
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=16
)
train_loader.sampler.generator.manual_seed(42)
random.seed(42)

# We make 123 positive samples, 240-123=117 negative samples (while we have 160 positive images and 80 negative images)
print("Number of positive samples in the training dataset: ", torch.tensor([t[1].sum() for t in iter(train_loader)]).sum())

# ## Put all together
# x_tensor = torch.as_tensor(images/255).float()
# y_tensor = torch.as_tensor(labels.reshape(-1, 1)).float() # [300,1]
# class TransformedTensorDataset(Dataset):
#     def __init__(self, x, y, transform=None):
#         self.x = x 
#         self.y = y
#         self.transform = transform 
#     def __getitem__(self, index):
#         x = self.x[index]
#         if self.transform:
#             x = self.transform(x)
#         return x, self.y[index]
#     def __len__(self):
#         return len(self.x)
# def index_splitter(n, splits, seed=13):
#     idx = torch.arange(n)
#     splits_tensor = torch.as_tensor(splits)
#     total = splits_tensor.sum().float()
#     # if total does not add up to 1.0, make the splits adding up to 1.0 e.g. [0.8, 0.2]
#     if not total.isclose(torch.ones(1)[0]):
#         splits_tensor = splits_tensor/total 
#     torch.manual_seed(seed)
#     return random_split(idx, splits_tensor)
# train_idx, val_idx = index_splitter(len(x_tensor), [80, 20])
# train_composer = Compose([
#     RandomHorizontalFlip(p=0.5),
#     Normalize(mean=(0.5,), std=(0.5,))
# ])
# val_composer = Compose([
#     Normalize(mean=(0.5,), std=(0.5,))
# ])
# train_dataset = TransformedTensorDataset(
#     x_train_tensor, y_train_tensor, transform=train_composer
# )
# val_dataset = TransformedTensorDataset(
#     x_val_tensor, y_val_tensor, transform=val_composer
# )
# sampler = make_balanced_sampler(y_train_tensor)
# train_loader = DataLoader(
#     dataset=train_dataset, batch_size=16, sampler=sampler
# )
# val_loader = DataLoader(
#     dataset=val_dataset, batch_size=16
# )

## Flatten to use pixels as features
dummy_xs, dummy_ys = next(iter(train_loader))
print("Dummy data shape: ", dummy_xs.shape)
flattener = nn.Flatten()
dummy_xs_flat = flattener(dummy_xs)
print("Dummy flattened data shape: ", dummy_xs_flat.shape)
print("1st dummy flattened data value: ", dummy_xs_flat[0])
lr = 0.1
torch.manual_seed(17)
model_logistic = nn.Sequential()
model_logistic.add_module('flatten', nn.Flatten())
model_logistic.add_module('output', nn.Linear(25, 1, bias=False))
model_logistic.add_module('sigmoid', nn.Sigmoid())
optimizer_logistic = optim.SGD(model_logistic.parameters(), lr=lr)
binary_loss_fn = nn.BCELoss()
n_epochs = 100
sbs_logistic = StepByStep(
    model_logistic, binary_loss_fn, optimizer_logistic)
sbs_logistic.set_loaders(train_loader, val_loader)
sbs_logistic.train(n_epochs)
fig = sbs_logistic.plot_losses()

## Try deeper model
lr = 0.1
torch.manual_seed(17)
model_nn = nn.Sequential()
model_nn.add_module('flatten', nn.Flatten())
model_nn.add_module('hidden0', nn.Linear(25, 5, bias=False))
model_nn.add_module('hidden1', nn.Linear(5, 3, bias=False))
model_nn.add_module('output', nn.Linear(3, 1, bias=False))
model_nn.add_module('sigmoid', nn.Sigmoid())
optimizer_nn = optim.SGD(model_nn.parameters(), lr=lr)
binary_loss_fn = nn.BCELoss()
n_epochs = 100
sbs_nn = StepByStep(model_nn, binary_loss_fn, optimizer_nn)
sbs_nn.set_loaders(train_loader, val_loader)
sbs_nn.train(n_epochs)
fig = sbs_nn.plot_losses()
fig = figure5(sbs_logistic, sbs_nn)

## Show that: deep model with no activation == shallow model with no hidden
w_nn_hidden0 = model_nn.hidden0.weight.detach()
w_nn_hidden1 = model_nn.hidden1.weight.detach()
w_nn_output = model_nn.output.weight.detach()
print('Deep weights shape: ', w_nn_hidden0.shape, w_nn_hidden1.shape, w_nn_output.shape)
# Another way for matrix multiplication
w_nn_equiv = w_nn_output.mm(w_nn_hidden1.mm(w_nn_hidden0))
w_nn_equiv = w_nn_output @ w_nn_hidden1 @ w_nn_hidden0
print('Equivalent deep weight shape: ', w_nn_equiv.shape)
w_logistic_output = model_logistic.output.weight.detach()
print('Shallow model weight shape: ', w_logistic_output.shape)
fig = weights_comparison(w_logistic_output, w_nn_equiv)

## Number of parameters of the two models
print("Number of parameters of the two models: ", sbs_logistic.count_parameters(), sbs_nn.count_parameters())

## Checking weights of the first layer as images
fig = figure7(w_nn_hidden0)
fig = plot_activation(torch.sigmoid)