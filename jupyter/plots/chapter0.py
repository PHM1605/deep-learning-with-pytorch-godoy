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
    surf = ax1.plot_surface(bs, ws, all_losses, rstride=1, cstride=1, alpha=0.5, cmap=plt.cm.jet, linewidth=0, antialiased=True)
    ax1.contour(bs[0,:], ws[:,0], all_losses, 10, offset=-1, cmap=plt.cm.jet)
    # find minimum location
    bidx, widx, _, _ = find_index(b_minimum, w_minimum, bs, ws)
    ax1.scatter(b_minimum, w_minimum, all_losses[bidx, widx], c='k')
    ax1.text(0.3, 1.5, all_losses[bidx, widx], "Minimum", zdir=(1,0,0))
    # Random start point
    bidx, widx, _, _ = find_index(b, w, bs, ws)
    ax1.scatter(b, w, all_losses[bidx, widx], c='k')
    ax1.text(0, -0.5, all_losses[bidx, widx], 'Random\n Start', zdir=(1,0,0))
    ax1.view_init(40, 260)

    ax2 = figure.add_subplot(1,2,2)
    ax2.set_xlabel('b')
    ax2.set_ylabel('w')
    ax2.set_title('Loss Surface')
    CS = ax2.contour(bs[0,:], ws[:,0], all_losses, cmap=plt.cm.jet)
    ax2.clabel(CS, inline=1, fontsize=10)
    ax2.scatter(b_minimum, w_minimum, c='k')
    ax2.scatter(b, w, c='k')
    ax2.annotate('Random Start', xy=(-0.2,0.05), c='k')
    ax2.annotate("Minimum", xy=(0.5,2.2), c="k")
    
    figure.tight_layout()
    return figure, (ax1)

def figure5(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train)
    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)
    b_range = bs[0,:]
    w_range = ws[:,0]
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].set_title("Loss Surface")
    axs[0].set_xlabel("b")
    axs[0].set_ylabel("w")
    # Loss surface 
    CS = axs[0].contour(bs[0,:], ws[:,0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    axs[0].scatter(b_minimum, w_minimum, c="k")
    axs[0].scatter(fixedb, fixedw, c="k")
    # Vertical section
    axs[0].plot([fixedb, fixedb], w_range[[0,-1]], linestyle="--", c='r', linewidth=2)
    # Annotation
    axs[0].annotate("Minimum", xy=(0.5,2.2), c="k")
    axs[0].annotate("Random Start", xy=(fixedb+0.1, fixedw+0.1), c="k")

    # 1d plot
    axs[1].set_title("Fixed: b={:.2f}".format(fixedb))
    axs[1].set_ylim([-0.1, 15.1])
    axs[1].set_xlabel('w')
    axs[1].set_ylabel('Loss')
    axs[1].plot(w_range, all_losses[:,b_idx], c='r', linestyle="--", linewidth=2)
    # Starting point
    axs[1].plot([fixedw], [all_losses[w_idx, b_idx]], "or")
    
    fig.tight_layout()
    return fig, axs
    
