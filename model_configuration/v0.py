import torch 
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.1
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(1, 1)).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean')