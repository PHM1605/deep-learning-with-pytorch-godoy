import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

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