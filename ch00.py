import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Synthetic data generate
true_b = 1
true_w = 2
N = 100
np.random.seed(42)
x = np.random.rand(N, 1) # N random numbers from 0 to 1
epsilon = (0.1 * np.random.randn(N, 1)) # N random Gaussian with mean 0 and variance 1
y = true_b + true_w * x + epsilon

# Train-Validation-Test Split
idx = np.arange(N)
np.random.shuffle(idx)
train_idx = idx[:int(N*.8)]
val_idx = idx[int(N*.8):]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = y[val_idx], y[val_idx]