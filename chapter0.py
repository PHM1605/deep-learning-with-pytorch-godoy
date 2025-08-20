import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from plots.chapter0 import *

true_b = 1 
true_w = 2
N = 100
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (0.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon 

# train-test split
idx = np.arange(N)
np.random.shuffle(idx)
train_idx = idx[:int(N*0.8)]
val_idx = idx[int(N*.8):]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
# figure1(x_train, y_train, x_val, y_val)

# init values for b and w gues
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)
yhat = b + w * x_train
# figure2(x_train, y_train, b, w)

# Loss of one (first) point
# figure3(x_train, y_train, b, w)

# Loss of all points
error = (yhat - y_train)
loss = (error**2).mean()
print("Loss: ", loss)

# Loss surface
b_range = np.linspace(true_b-3, true_b+3, 101)
w_range = np.linspace(true_w-3, true_w+3, 101)
bs, ws = np.meshgrid(b_range, w_range)
print(bs.shape, ws.shape) # [101, 101]; [101, 101]

# Pick a value of x, calculate the prediction value y from that x-value for all possible combinations of w and b
dummy_x = x_train[0] # [1]
dummy_yhat = bs + ws * dummy_x # broadcasting [1] and [101, 101] to [101, 101]
print(dummy_yhat.shape) # [101, 101] 

# Do this for all values of x
all_predictions = np.apply_along_axis(
  func1d = lambda x: bs + ws * x, 
  axis = 1,
  arr = x_train # [80, 1]
)
print(all_predictions.shape) # [80, 101, 101]
all_labels = y_train.reshape(-1, 1, 1) # [80, 1] -> [80, 1, 1]
print(all_labels.shape) # [80, 1, 1]
all_errors = (all_predictions - all_labels)
print(all_errors.shape) # [80, 101, 101]
all_losses = (all_errors ** 2).mean(axis=0)
print(all_losses.shape) # [101, 101]

# plot surf 
# figure4(x_train, y_train, b, w, bs, ws, all_losses)

# plot contour when fixing b and varying w
# figure5(x_train, y_train, b, w, bs, ws, all_losses)

# plot contour when fixing w and varying b
# figure6(x_train, y_train, b, w, bs, ws, all_losses)

## Compute gradients for both b- and w-parameters
b_grad = 2 * error.mean()
w_grad = 2 * (x_train * error).mean()
print("B_GRAD:", b_grad, "; W_GRAD:", w_grad)
# Visualizing gradients
# figure7(b, w, bs, ws, all_losses)

# Visualizing gradients - zooming in
# figure8(b, w, bs, ws, all_losses)

## Backpropagation
# Update the parameters
lr = .1
print("B BEFORE: " + str(b) + "; W BEFORE: " + str(w))
b = b - lr * b_grad
w = w - lr * w_grad 
print("B AFTER: " + str(b) + "; W AFTER: " + str(w))
figure9(x_train, y_train, b, w)

manual_grad_b = -2.9
manual_grad_w = -1.79
np.random.seed(42)
b_initial = np.random.randn(1)
w_initial = np.random.randn(1)
lr = .2
#figure10(b_initial, w_initial, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr)
lr = 1.1 # too high learning rate
# figure10(b_initial, w_initial, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr)

# Normalization necessity illustration
true_b = 1
true_w = 2
N = 100
np.random.seed(42)
bad_w = true_w / 10 
bad_x = np.random.rand(N, 1) * 10 
y = true_b + bad_w * bad_x + (.1 * np.random.randn(N, 1)) # more or less will be the same even with bad samples
bad_x_train, y_train = bad_x[train_idx], y[train_idx]
bad_x_val, y_val = bad_x[val_idx], y[val_idx]
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(x_train, y_train)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_ylim([0, 3.1])
ax[0].set_title('Train  - Original')
ax[1].scatter(bad_x_train, y_train, c='k')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_ylim([0, 3.1])
ax[1].set_title('Train - "Bad"')
fig.tight_layout()
bad_b_range = np.linspace(-2, 4, 101)
bad_w_range = np.linspace(-2.8, 3.2, 101)
bad_bs, bad_ws = np.meshgrid(bad_b_range, bad_w_range)
# figure14(x_train, y_train, b_initial, w_initial, bad_bs, bad_ws, bad_x_train)
# figure15(x_train, y_train, b_initial, w_initial, bad_bs, bad_ws, bad_x_train)

## Normalization/ standardization
# Scatter plot
scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(x_train)
scaled_x_train = scaler.transform(x_train)
scaled_x_val = scaler.transform(x_val)
fig, ax = plt.subplots(1, 3, figsize=(15,6))
ax[0].scatter(x_train, y_train, c='b')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_ylim([0, 3.1])
ax[0].set_title('Train - Original')
ax[1].scatter(bad_x_train, y_train, c='k')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_ylim([0, 3.1])
ax[1].set_title('Train - "Bad"')
ax[1].label_outer()
ax[2].scatter(scaled_x_train, y_train, c='g')
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')
ax[2].set_ylim([0, 3.1])
ax[2].set_title('Train - Scaled')
ax[2].label_outer()
fig.tight_layout()
# Loss contour
scaled_b_range = np.linspace(-1, 5, 101)
scaled_w_range = np.linspace(-2.4, 3.6, 101)
scaled_bs, scaled_ws = np.meshgrid(scaled_b_range, scaled_w_range)
figure17(x_train, y_train, scaled_bs, scaled_ws, bad_x_train, scaled_x_train)
figure18(x_train, y_train)