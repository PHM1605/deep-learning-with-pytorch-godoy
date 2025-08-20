import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

def fit_model(x_train, y_train):
    regression = LinearRegression()
    regression.fit(x_train, y_train)
    b_minimum, w_minimum = regression.intercept_[0], regression.coef_[0][0]
    return b_minimum, w_minimum

def figure1(x_train, y_train, x_val, y_val):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
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

def figure3(x_train, y_train):
    b_minimum, w_minimum = fit_model(x_train, y_train)
    x_range = np.linspace(0, 1, 101)
    yhat_range = b_minimum + w_minimum * x_range
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3.1])
    ax.scatter(x_train, y_train)
    ax.plot(x_range, yhat_range, label='Final mode\'s predictions', c='k', linestyle='--')
    ax.annotate('b={:.4f} w={:.4f}'.format(b_minimum, w_minimum), xy=(0.4, 1.4), c='k', rotation=34)
    ax.legend(loc=0)
    fig.tight_layout()
    return fig, ax