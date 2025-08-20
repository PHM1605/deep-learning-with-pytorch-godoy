import numpy as np
from PIL import Image 
from copy import deepcopy
import torch 
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms.v2 import Compose, ToImage, Normalize, ToPILImage, Resize, ToDtype
from torchvision.datasets import ImageFolder 
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, CyclicLR, LambdaLR 
from stepbystep.v3 import StepByStep 
from data_generation.rps import download_rps 
from plots.chapter6 import *
from torch_lr_finder import LRFinder
from data_generation.simple_linear_regression import lr_data_generate
from data_preparation.v2 import prepare_data

# download_rps()
# fig = figure1()

# # Pilllow image: RGB
# temp_transform = Compose([Resize(28), ToImage(), ToDtype(torch.float32, scale=True)])
# temp_dataset = ImageFolder(root='rps', transform=temp_transform)
# print("X, y of 1 sample: ", temp_dataset[0][0].shape, temp_dataset[0][1])
# temp_loader = DataLoader(temp_dataset, batch_size=16)

# ## Check statistics for one batch
# first_images, first_labels = next(iter(temp_loader))
# print("Statistics per channel: ", StepByStep.statistics_per_channel(first_images, first_labels))
# ## Check statistics for all batches
# results = StepByStep.loader_apply(temp_loader, StepByStep.statistics_per_channel)
# print("Sum of statistics of all batches: ", results)
# normalizer = StepByStep.make_normalizer(temp_loader)
# print("Normalizer: ", normalizer)

# ## The real dataset 
# composer = Compose([
#     Resize(28),
#     ToImage(),
#     ToDtype(torch.float32, scale=True),
#     normalizer
# ])
# train_data = ImageFolder(root='rps', transform=composer)
# val_data = ImageFolder(root='rps-test-set', transform=composer)
# train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=16)
# torch.manual_seed(88)
# first_images, first_labels = next(iter(train_loader))
# fig = figure2(first_images, first_labels)

class CNN2(nn.Module):
    def __init__(self, n_filters, p=0.0):
        super(CNN2, self).__init__()
        self.n_filters = n_filters
        self.p = p
        # Featurizer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_filters, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3)
        # Classifier
        self.fc1 = nn.Linear(n_filters*5*5, 50)
        self.fc2 = nn.Linear(50, 3)
        self.drop = nn.Dropout(self.p)

    def featurizer(self, x):
        x = self.conv1(x) # [3,28,28]->[n_filters,26,26]
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # [n_filters,13,13]
        x = self.conv2(x) # [n_filters,11,11]
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # [n_filters,5,5]
        x = nn.Flatten()(x)
        return x 
    
    def classifier(self, x):
        if self.p > 0:
            x = self.drop(x) # [n_filters*5*5]
        x = self.fc1(x) # [50]
        x = F.relu(x)
        if self.p > 0:
            x = self.drop(x)
        x = self.fc2(x)
        return x 
    
    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)
        return x 

# ## Dropout
# dropping_model = nn.Sequential(nn.Dropout(p=0.5))
# spaced_points = torch.linspace(0.1, 1.1, 11)
# print("Input spaced points: ", spaced_points)
# torch.manual_seed(44)
# dropping_model.train()
# output_train = dropping_model(spaced_points)
# # Some outputs are demolished to 0; the rest are scaled by 1/p (in this case, multiplying by 2)
# print("Output of dropping-model: ", output_train)
# print("Weight of adjusted output: ", F.linear(output_train, weight=torch.ones(11), bias=torch.tensor(0))) # 9.4
# # Why need adjusting? Because there is no dropout in 'eval' mode
# dropping_model.eval()
# output_eval = dropping_model(spaced_points)
# print("Output of non-dropping (eval) model: ", output_eval)
# print("Weight of non-adjusted eval output: ", F.linear(output_eval, weight=torch.ones(11), bias=torch.tensor(0))) # 6.6
# # Run the dropout experiments 1000 times, take the value of adjusted dropout outputs -> calculate their weight-sum, we will see the mean of those 1000 samples is closed to 6.6
# torch.manual_seed(17)
# p = 0.5
# distrib_outputs = torch.tensor([
#     F.linear(F.dropout(spaced_points, p=p), weight=torch.ones(11), bias=torch.tensor(0))
#     for _ in range(1000)
# ])
# fig = figure7(p, distrib_outputs)
# fig = figure8()

# ## Dropout2D - dropping 'channels' instead of 'pixels'
# fig = figure9(first_images)

# ## Model config (without dropout)
# torch.manual_seed(13)
# model_cnn2 = CNN2(n_filters=5, p=0.3)
# multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
# optimizer_cnn2 = optim.Adam(model_cnn2.parameters(), lr=3e-4)
# sbs_cnn2 = StepByStep(model_cnn2, multi_loss_fn, optimizer_cnn2)
# sbs_cnn2.set_loaders(train_loader, val_loader)
# sbs_cnn2.train(10)
# fig = sbs_cnn2.plot_losses()
# print(StepByStep.loader_apply(val_loader, sbs_cnn2.correct))

# ## Model config (with dropout)
# torch.manual_seed(13)
# model_cnn2_nodrop= CNN2(n_filters=5, p=0.0)
# multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
# optimizer_cnn2_nodrop = optim.Adam(model_cnn2_nodrop.parameters(), lr=3e-4)
# sbs_cnn2_nodrop = StepByStep(model_cnn2_nodrop, multi_loss_fn, optimizer_cnn2_nodrop)
# sbs_cnn2_nodrop.set_loaders(train_loader, val_loader)
# sbs_cnn2_nodrop.train(10)

# ## Compare with dropout and without dropout
# fig = figure11(sbs_cnn2.losses, sbs_cnn2.val_losses, sbs_cnn2_nodrop.losses, sbs_cnn2_nodrop.val_losses)
# print("Correct/Total result for no-dropout-model, 1st is training result, 2nd is validation result: ", 
#       StepByStep.loader_apply(train_loader, sbs_cnn2_nodrop.correct).sum(axis=0),
#       StepByStep.loader_apply(val_loader, sbs_cnn2_nodrop.correct).sum(axis=0)
#       )
# print("Correct/Total result for with-dropout-model, 1st is training result, 2nd is validation result: ",
#       StepByStep.loader_apply(train_loader, sbs_cnn2.correct).sum(axis=0),
#       StepByStep.loader_apply(val_loader, sbs_cnn2.correct).sum(axis=0)
#       )

# ## Visualizing Filters
# print("Shape of 1st ConvLayer of DropoutModel: ", model_cnn2.conv1.weight.shape) # [num_filters,num_channels,height,width]=[5,3,3,3]
# fig = sbs_cnn2.visualize_filters('conv1')
# print("Shape of 2nd ConvLayer of DropoutModel: ", model_cnn2.conv2.weight.shape) # [num_filters,num_channels,height,width]=[5,5,3,3]
# fig = sbs_cnn2.visualize_filters('conv2')

# ## Range of learning rates testing on DummyModel
# def make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp'):
#     # iteration (list): [0,1,2,...10]
#     if step_mode == 'linear':
#         factor = (end_lr/start_lr - 1) / num_iter 
#         def lr_fn(iteration):
#             return 1 + iteration * factor
#     else:
#         factor = (np.log(end_lr) - np.log(start_lr)) / num_iter
#         def lr_fn(iteration):
#             return np.exp(factor)**iteration 
#     return lr_fn 
# start_lr = 0.01
# end_lr = 0.1
# num_iter = 10
# lr_fn = make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp')
# print("Learning rate list: ", start_lr * lr_fn(np.arange(num_iter+1))) # [0.01, 0.0123, ...., 0.1]
# dummy_model = CNN2(n_filters=5, p=0.3)
# dummy_optimizer = optim.Adam(dummy_model.parameters(), lr=start_lr)
# dummy_scheduler = LambdaLR(dummy_optimizer, lr_lambda=lr_fn)
# dummy_optimizer.step()
# dummy_scheduler.step()
# print("Scheduler current learning rate: ", dummy_scheduler.get_last_lr()[0])

# ## Range of learning rates testing on DropoutModel
# torch.manual_seed(13)
# new_model = CNN2(n_filters=5, p=0.3)
# multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
# new_optimizer = optim.Adam(new_model.parameters(), lr=3e-4)
# sbs_new = StepByStep(new_model, multi_loss_fn, new_optimizer)
# tracking, fig = sbs_new.lr_range_test(train_loader, end_lr=0.1, num_iter=100) # choose the learning rate at the inflection point of the U-curve => 0.005

# ## Set the new optimal learning rate and check the curves
# new_optimizer = optim.Adam(new_model.parameters(), lr=0.005)
# sbs_new.set_optimizer(new_optimizer)
# sbs_new.set_loaders(train_loader, val_loader)
# sbs_new.train(10)
# fig = sbs_new.plot_losses() # training loss actually goes down faster  

# ## LRFinder
# torch.manual_seed(11)
# new_model = CNN2(n_filters=5, p=0.3)
# multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
# new_optimizer = optim.Adam(new_model.parameters(), lr=3e-4)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# lr_finder = LRFinder(
#     new_model, new_optimizer, multi_loss_fn, device=device
# )
# lr_finder.range_test(train_loader, end_lr=1e-1, num_iter=100)
# lr_finder.plot(log_lr=True)
# lr_finder.reset()

# ## EWMA
# fig = figure15()
# # To prove: average-age-of-ewma = alpha*sum-over-lag-from-0-to-T-minus-1((1-alpha)**lag*(lag+1)) == 1/alpha
# alpha = 1/3
# T = 20
# t = np.arange(1, T+1)
# age = alpha * sum((1-alpha)**(t-1)*t)
# print("Age of EWMA: ", age)
# # Calculation
# temperatures = np.array([5, 11, 15, 6, 5, 3, 3, 0, 0, 3, 4, 2, 1, -1, -2, 2, 2, -2, -1, -1, 3, 4, -1, 2, 6, 4, 9, 11, 9, -2])
# fig = ma_vs_ewma(temperatures, periods=19)

# ## Adam optimizer
# optimizer = optim.Adam(model_cnn2.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)
# x, y, x_train, y_train, x_val, y_val = lr_data_generate()
# x_tensor, y_tensor, train_data, val_data, train_loader, val_loader = prepare_data(x, y)

# ## Prepare model for gradients capturing
# torch.manual_seed(42)
# model = nn.Sequential()
# model.add_module('linear', nn.Linear(1,1))
# optimizer = optim.Adam(model.parameters(), lr=0.1)
# loss_fn = nn.MSELoss(reduction='mean')
# sbs_adam = StepByStep(model, loss_fn, optimizer)
# sbs_adam.set_loaders(train_loader)
# sbs_adam.capture_gradients('linear')
# sbs_adam.train(10) # after training we will get {'linear': {'weight': [50 elements], 'bias':[50 elements]}} as each epoch of 10 training-epochs has 5 batches
# sbs_adam.remove_hooks()
# gradients = np.array(sbs_adam._gradients['linear']['weight']).squeeze()
# corrected_gradients = calc_corrected_ewma(gradients, 19)
# corrected_sq_gradients = calc_corrected_ewma(np.power(gradients, 2), 1999)
# adapted_gradients = corrected_gradients/ (np.sqrt(corrected_sq_gradients) + 1e-8)
# fig = figure17(gradients, corrected_gradients, corrected_sq_gradients, adapted_gradients)
# # Optimizer will store last-ewma (no bias-corrected) of 'weight' & last-ewma (no bias-corrected) of 'bias'
# print("Optimizer state dict: ", optimizer.state_dict())
# print("Manual calculated: ", calc_ewma(gradients, 19)[-1], calc_ewma(np.power(gradients,2), 1999)[-1])

# b, w, bs, ws, all_losses = contour_data(x_tensor, y_tensor)

# ## Viewing weight progression for SGD and Adam
# torch.manual_seed(42)
# model = nn.Sequential()
# model.add_module('linear', nn.Linear(1, 1))
# loss_fn = nn.MSELoss(reduction='mean')
# optimizers = {
#     'SGD': {'class': optim.SGD, 'parms': {'lr': 0.1}},
#     'Adam': {'class': optim.Adam, 'parms': {'lr': 0.1}}
#     }
# results = compare_optimizers(model, loss_fn, optimizers, train_loader, val_loader, n_epochs=10)
# fig = plot_paths(results, b, w, bs, ws, all_losses)
# fig = plot_losses(results)

# ## Momentum: momentum_t = beta*momentum_(t-1) + grad_t = grad_t + beta*grad_(t-1) + beta^2*grad_(t-2)+....
# # Momentum with dampening factor: momentum_t = beta*momentum_(t-1) + (1-damp)*grad_t
# # => param_t = param_(t-1) - eta*momentum_t
# torch.manual_seed(42)
# model = nn.Sequential()
# model.add_module('linear', nn.Linear(1,1))
# loss_fn = nn.MSELoss(reduction='mean')
# optimizers = {
#     'SGD': {'class': optim.SGD, 'parms': {'lr':0.1}},
#     'SGD+Momentum': {'class': optim.SGD, 'parms': {'lr':0.1, 'momentum':0.9}}
# }
# results = compare_optimizers(model, loss_fn, optimizers, train_loader, val_loader, n_epochs=10)
# print("SGD+Momentum state dict: ", results['SGD+Momentum']['state'])
# fig = plot_paths(results, b, w, bs, ws, all_losses)

# ## Nesterov accelerated gradient
# # momentum_t = beta*momentum_(t-1) + grad_t
# # nesterov_t = beta*momentum_t + grad_t 
# # => param_t = param_(t-1) - eta*nesterov_t = param_(t-1) - eta*grad_t - eta*beta*momentum_t
# torch.manual_seed(42)
# model = nn.Sequential()
# model.add_module('linear', nn.Linear(1, 1))
# loss_fn = nn.MSELoss(reduction='mean')
# optimizers = {
#     'SGD': {'class': optim.SGD, 'parms': {'lr': 0.1}},
#     'SGD+Momentum': {'class': optim.SGD, 'parms': {'lr':0.1, 'momentum':0.9}},
#     'SGD+Nesterov': {'class': optim.SGD, 'parms': {'lr':0.1, 'momentum':0.9, 'nesterov':True}}
# }
# results = compare_optimizers(model, loss_fn, optimizers, train_loader, val_loader, n_epochs=10)
# # gradients, momentums and nesterovs
# fig = figure21(results)
# # loss contour and weight map update
# fig = plot_paths(results, b, w, bs, ws, all_losses)
# # loss curve over 10 epochs
# fig = plot_losses(results)

# ## Learning Rate Schedulers
# # Scheduler multiplies 0.1 every 2 epochs
# dummy_optimizer = optim.SGD([nn.Parameter(torch.randn(1))], lr=0.01)
# dummy_scheduler = StepLR(dummy_optimizer, step_size=2, gamma=0.1)
# print("Learning rate every 2 epochs: ")
# for epoch in range(4):
#     print(dummy_scheduler.get_last_lr())
#     dummy_optimizer.step()
#     dummy_scheduler.step()
#     dummy_optimizer.zero_grad()
# # Scheduler sets the learning rate based on a function
# # this is the same as StepLR(dummy_optimizer, step_size=2, gamma=0.1)
# dummy_optimizer = optim.SGD([nn.Parameter(torch.randn(1))], lr=0.01)
# dummy_scheduler = LambdaLR(
#     dummy_optimizer, lr_lambda = lambda epoch: 0.1**(epoch//2)
# )
# fig = plot_scheduler(dummy_optimizer, dummy_scheduler)
# # Scheduler based on not-good-enough improvement over 'patience' epochs in validation loss
# dummy_optimizer = optim.SGD([nn.Parameter(torch.randn(1))], lr=0.01)
# dummy_scheduler = ReduceLROnPlateau(dummy_optimizer,patience=4, factor=0.1)
# fig = plot_scheduler(dummy_optimizer, dummy_scheduler)
# # Mini-batch Scheduler
# dummy_parm = [nn.Parameter(torch.randn(1))]
# dummy_optimizer = optim.SGD(dummy_parm, lr=0.01)
# dummy_scheduler1 = CyclicLR(dummy_optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=2, mode='triangular')
# dummy_scheduler2 = CyclicLR(dummy_optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=2, mode='triangular2') # peak of trianglles halving every 2 batches
# dummy_scheduler3 = CyclicLR(dummy_optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=2, mode='exp_range', gamma=np.sqrt(0.5)) # peak of triangles are reduced exponentially every 2 batches
# fig = figure26(dummy_optimizer, (dummy_scheduler1, dummy_scheduler2, dummy_scheduler3))

# ## LR Range Test to find the best LR
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# for ax, nesterov in zip(axs.flat, [False, True]):
#     torch.manual_seed(42)
#     model = nn.Sequential()
#     model.add_module('linear', nn.Linear(1, 1))
#     loss_fn = nn.MSELoss(reduction='mean')
#     optimizer = optim.SGD(
#         model.parameters(), lr=1e-3, momentum=0.9, nesterov=nesterov
#     )
#     sbs_scheduler = StepByStep(model, loss_fn, optimizer)
#     tracking, fig = sbs_scheduler.lr_range_test(train_loader, end_lr=1, num_iter=100, ax=ax)
#     nest = ' + Nesterov' if nesterov else ''
#     ax.set_title(f'Momentum{nest}')

# ## Combining all gradients-with-momentum+nesterov and learning-rate-scheduler WHEN training weights
# # from the above curves we choose the elbow point as optimal base learning-rate i.e. 0.025
# step_scheduler = StepLR(optimizer, step_size=4, gamma=0.5) # per-epoch-scheduler: halving learning rate after 4 epochs
# cyclic_scheduler = CyclicLR(
#     optimizer, base_lr=0.025, max_lr=1, step_size_up=10, mode='triangular2'
# ) # per-batch-scheduler
# torch.manual_seed(42)
# model = nn.Sequential()
# model.add_module('linear', nn.Linear(1,1))
# loss_fn = nn.MSELoss(reduction='mean')
# optimizers = {
#     'SGD+Momentum': {'class':optim.SGD, 'parms': {'lr':0.1, 'momentum':0.9}},
#     'SGD+Momentum+Step':{'class':optim.SGD, 'parms':{'lr':0.1, 'momentum':0.9}},
#     'SGD+Momentum+Cycle':{'class':optim.SGD, 'parms':{'lr':0.1, 'momentum':0.9}},
#     'SGD+Nesterov': {'class':optim.SGD, 'parms':{'lr':0.1, 'momentum':0.9, 'nesterov':True}},
#     'SGD+Nesterov+Step':{'class':optim.SGD, 'parms':{'lr':0.1, 'momentum':0.9, 'nesterov':True}},
#     'SGD+Nesterov+Cycle':{'class':optim.SGD, 'parms':{'lr':0.1, 'momentum':0.9, 'nesterov':True}}
# }
# schedulers = {
#     'SGD+Momentum+Step': {'class':StepLR, 'parms': {'step_size':4, 'gamma':0.5}},
#     'SGD+Momentum+Cycle': {'class':CyclicLR, 'parms': {'base_lr':0.025, 'max_lr':0.1, 'step_size_up':10, 'mode':'triangular2'}},
#     'SGD+Nesterov+Step': {'class':StepLR, 'parms': {'step_size':4, 'gamma':0.5}},
#     'SGD+Nesterov+Cycle': {'class':CyclicLR, 'parms': {'base_lr':0.025, 'max_lr':0.1, 'step_size_up':10, 'mode':'triangular2'}}
# }
# results = compare_optimizers(model, loss_fn, optimizers, train_loader, val_loader, schedulers, n_epochs=10)
# fig = figure28(results, b, w, bs, ws, all_losses)
# # loss progression
# fig, axs = plt.subplots(2, 3, figsize=(15, 8))
# axs = axs.flat 
# fig = plot_losses(results, axs)

## Putting all together
temp_transform = Compose([Resize(28), ToImage(), ToDtype(torch.float32, scale=True)])
temp_dataset = ImageFolder(root='rps', transform=temp_transform)
temp_loader = DataLoader(temp_dataset, batch_size=16)
normalizer = StepByStep.make_normalizer(temp_loader)
composer = Compose([Resize(28), ToImage(), ToDtype(torch.float32, scale=True), normalizer])
train_data = ImageFolder(root='rps', transform=composer)
val_data = ImageFolder(root='rps-test-set', transform=composer)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
torch.manual_seed(13)
model_cnn3 = CNN2(n_filters=5, p=0.5)
multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer_cnn3 = optim.SGD(model_cnn3.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
sbs_cnn3 = StepByStep(model_cnn3, multi_loss_fn, optimizer_cnn3)
tracking, fig = sbs_cnn3.lr_range_test(train_loader, end_lr=2e-1, num_iter=100) 
# from the lr_range_test => we choose base learning rate at elbow point = 0.025
optimizer_cnn3 = optim.SGD(model_cnn3.parameters(), lr=0.025, momentum=0.9, nesterov=True)
sbs_cnn3.set_optimizer(optimizer_cnn3)
# step_size_up = number of batches in 1 epoch means the cyclic learning rate period is 2 epochs
scheduler = CyclicLR(optimizer_cnn3, base_lr=1e-3, max_lr=0.025, step_size_up=len(train_loader), mode='triangular2')
sbs_cnn3.set_lr_scheduler(scheduler)
sbs_cnn3.set_loaders(train_loader, val_loader)
sbs_cnn3.train(10)
fig = sbs_cnn3.plot_losses()
print("Training and validation recall: ")
print(
    StepByStep.loader_apply(train_loader, sbs_cnn3.correct).sum(axis=0),
    StepByStep.loader_apply(val_loader, sbs_cnn3.correct).sum(axis=0)
)