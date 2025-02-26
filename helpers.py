import numpy as np 
import torch
from torch.utils.data import random_split, WeightedRandomSampler

# splits = [80,20]
def index_splitter(n, splits, seed=13):
    idx = torch.arange(n)
    splits_tensor = torch.as_tensor(splits)
    total = splits_tensor.sum().float()
    # [80,20]->[0.8,0.2]
    if not total.isclose(torch.ones(1)[0]):
        splits_tensor = splits_tensor / total 
    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)

def make_balanced_sampler(y):
    classes, counts = y.unique(return_counts=True)
    weights = 1.0/counts.float()
    sample_weights = weights[y.squeeze().long()]
    generator = torch.Generator()
    sampler = WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(sample_weights),
        generator = generator,
        replacement = True
    )
    return sampler 
