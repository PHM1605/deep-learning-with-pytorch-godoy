import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

plt.style.use('fivethirtyeight')

# fits a linear regression to find the actual b and w that minimize the loss 
def fit_model(x_train, y_train):
    regression = LinearRegression()
    regression.fit(x_train, y_train)
    b_minimum, w_minimum = regression.intercept_[0], regression.coef_[0][0]
    return b_minimum, w_minimum

# find the closest indexes for the updated b and w in their respective ranges
# bs: column has same elements; ws: row has same elements
def find_index(b, w, bs, ws):
    b_idx = np.argmin(np.abs(bs[0,:] - b))
    w_idx = np.argmin(np.abs(ws[:,0] - w))
    fixedb, fixedw = bs[0, b_idx], ws[w_idx, 0]
    return b_idx, w_idx, fixedb, fixedw

def figure1(x_train, y_train, x_val, y_val):
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].scatter(x_train, y_train)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_ylim([0, 3.1])
    ax[0].set_title('Generated Data - Train')

    ax[1].scatter(x_val, y_val, c='r')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_ylim([0, 3.1])
    ax[1].set_title('Generated Data - Validation')
    fig.tight_layout()

    return fig, ax

def figure2(x_train, y_train, b, w, color='k'):
    x_range = np.linspace(0, 1, 101)
    yhat_range = b + w * x_range 
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3])
    
    ax.scatter(x_train, y_train)
    ax.plot(x_range, yhat_range, label='Model\'s predictions', c=color, linestyle='--')
    ax.annotate('b={:.4f} w={:.4f}'.format(b[0], w[0]), xy=(0.2, 0.55), c=color)
    ax.legend(loc=0)
    fig.tight_layout()
    return fig, ax

def figure3(x_train, y_train, b, w):
    fig, ax = figure2(x_train, y_train, b, w)
    # First data point
    x0, y0 = x_train[0][0], y_train[0][0]
    ax.scatter([x0], [y0], c='r')
    # Vertical line showing error between point and prediction
    ax.plot([x0, x0], [b[0]+w[0]*x0, y0-0.03], c='r', linewidth=2, linestyle='--')
    ax.arrow(x0, y0-0.03, 0, 0.03, color='r', shape='full', lw=0, length_includes_head=True, head_width=0.03)
    ax.arrow(x0, b[0]+w[0]*x0+0.05, 0, -0.03, color='r', shape='full', lw=0, length_includes_head=True, head_width=0.03)
    # Annotate
    ax.annotate(r'$error_0$', xy=(0.8, 1.5))
    fig.tight_layout()
    return fig, ax

def figure4(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train) 
    figure = plt.figure(figsize=(12,6))
    # 1st plot
    ax1 = figure.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('b')
    ax1.set_ylabel('w')
    ax1.set_title('Loss Surface')
    surf = ax1.plot_surface(bs, ws, all_losses, rstride=1, cstride=1, alpha=0.5, cmap=plt.cm.jet, linewidth, antialiased=True)
    figure.tight_layout()
    return figure, (ax1)
    