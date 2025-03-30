from data_generation.ball import load_data
import torch 
import torch.optim as optim 
from sklearn.datasets import make_regression 
from torch.utils.data import DataLoader, TensorDataset 
from stepbystep.v3 import StepByStep
from plots.chapter_extra import *

# X, y = load_data(n_points=1000, n_dims=10) # [n_samples,n_dims], [n_samples,1]
# ball_dataset = TensorDataset(
#     torch.as_tensor(X).float(), torch.as_tensor(y).float()
# )
# ball_loader = DataLoader(ball_dataset, batch_size=len(X))
# torch.manual_seed(11)
# n_layers = 5
# n_features = X.shape[1]
# hidden_units = 100 
# activation_fn = nn.ReLU 
# model = build_model(n_features, n_layers, hidden_units, activation_fn, use_bn=False)
# print("Very deep model: ", model)
# parms, gradients, activations = get_plot_data(train_loader=ball_loader, model=model) # each is [[(1000),(1000),(1000),(1000),(1000)]]
# fig = plot_violins(parms, gradients, activations)

# ## Not used, only for information  
# # nn.Linear.reset_parameters
# def reset_parameters(self) -> None:
#     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#     if self.bias is not None:
#         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#         bound = 1/math.sqrt(fan_in)
#         init.uniform_(self.bias, -bound, bound)

# # if we want to apply Kaiming uniform initialization scheme on a model
# def weights_init(m): # m is Layer
#     if isinstance(m, nn.Linear):
#         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
# with torch.no_grad():
#     model.apply(weights_init)

# ## Comparing 3 schemes:
# # 1. sigmoid activation + normal initialization
# # 2. tanh activation + Xavier uniform initialization
# # 3. relu activation + Kaiming uniform initialization
# fig = plot_schemes(n_features, n_layers, hidden_units, ball_loader)

# ## Comparing init schemes with batch normalization
# fig = plot_scheme_bn(n_features, n_layers, hidden_units, ball_loader)

## Exploding gradients
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_reg = torch.as_tensor(X_reg).float()
y_reg = torch.as_tensor(y_reg).float().view(-1,1)
dataset = TensorDataset(X_reg, y_reg)
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
fig = distributions(X_reg, y_reg)
torch.manual_seed(11)
model = nn.Sequential()
model.add_module('fc1', nn.Linear(10, 15))
model.add_module('act1', nn.ReLU())
model.add_module('fc2', nn.Linear(15, 1))
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# sbs_reg = StepByStep(model, loss_fn, optimizer)
# sbs_reg.set_loaders(train_loader)
# sbs_reg.capture_gradients(['fc1'])
# sbs_reg.train(2)
# print("Loss of regression: ", sbs_reg.losses) # will notice this too big
# # each epoch has 32 batches-> 2 epochs have 64 batches -> 64 gradients 
# grads = np.array(sbs_reg._gradients['fc1']['weight']) # [64,15,10]
# print("Gradients mean: ", grads.mean(axis=(1,2))) # [64,]

# ## Solution: Gradient clipping
# torch.manual_seed(42)
# parm = nn.Parameter(torch.randn(2,1))
# fake_grads = torch.tensor([[2.5],[0.8]])
# parm.grad = fake_grads.clone()
# # Value Clipping
# nn.utils.clip_grad_value_(parm, clip_value=1.0)
# print("Clipped gradient of parameter: ", parm.grad.view(-1,))
# fig = compare_grads(fake_grads, parm.grad) # direction of gradient descent has been changed
# # if we want to attach hook for clipping manually (don't forget to remove hook when done)
# def clip_backprop(model, clip_value):
#     handles = []
#     for p in model.parameters():
#         if p.requires_grad:
#             func = lambda grad: torch.clamp(grad, -clip_value, clip_value)
#             handle = p.register_hook(func)
#             handles.append(handle)
#     return handles 

# # Norm Clipping (Gradient Scaling)
# parm.grad = fake_grads.clone() 
# nn.utils.clip_grad_norm_(parm, max_norm=1.0, norm_type=2)
# print("Original grads norm: ", fake_grads.norm(), "; norm-clipped grads: ", parm.grad.view(-1,), "; norm-clipped grads norm: ", parm.grad.norm())
# fig = compare_grads(fake_grads, parm.grad) # direction of gradient does not change

# to reset the gradients being exploded above 
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
# method 1: clipping during training
torch.manual_seed(42)
with torch.no_grad():
    model.apply(weights_init)
optimizer = optim.SGD(model.parameters(), lr=0.1)
sbs_reg_clip = StepByStep(model, loss_fn, optimizer)
sbs_reg_clip.set_loaders(train_loader)
sbs_reg_clip.set_clip_grad_value(1.0)
sbs_reg_clip.capture_gradients(['fc1'])
sbs_reg_clip.train(10)
sbs_reg_clip.remove_clip()
sbs_reg_clip.remove_hooks()
fig = sbs_reg_clip.plot_losses()
avg_grad = np.array(
    sbs_reg_clip._gradients['fc1']['weight']
    ).mean(axis=(1,2)) # grads shape: [320,15,10] as we trained 10 epochs, 32 batch => mean of 320 numbers
print("Range of average grad before clipping: ", avg_grad.min(), avg_grad.max())
# method 2: clipping with hooks 
torch.manual_seed(42)
with torch.no_grad():
    model.apply(weights_init)
sbs_reg_clip_hook = StepByStep(model, loss_fn, optimizer)
sbs_reg_clip_hook.set_loaders(train_loader)
sbs_reg_clip_hook.set_clip_backprop(1.0)
sbs_reg_clip_hook.capture_gradients(['fc1'])
sbs_reg_clip_hook.train(10)
sbs_reg_clip_hook.remove_clip()
sbs_reg_clip_hook.remove_hooks()
fig = sbs_reg_clip_hook.plot_losses()
# compare gradients distribution of the two methods
fig = gradient_distrib(sbs_reg_clip, 'fc1', sbs_reg_clip_hook, 'fc1')