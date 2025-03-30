## Ball dataset: generate many points from a 10D ball (sphere) => zero mean and standard deviation
# points within half of radius => negative; otherwise positive
import numpy as np 

def load_data(n_dims=10, n_points=1000, classif_radius_fraction=0.5, only_sphere=False, shuffle=True, seed=13):
    np.random.seed(seed)
    radius = np.sqrt(n_dims)
    points = np.random.normal(size=(n_points, n_dims))
    sphere = radius * points / np.linalg.norm(points, axis=1).reshape(-1,1)
    if only_sphere:
        X = sphere 
    else:
        X = sphere * np.random.uniform(size=(n_points,1))**(1/n_dims)
    adjustment = 1/np.std(X)
    radius *= adjustment 
    X *= adjustment 
    y = (np.abs(np.sum(X, axis=1)) > (radius*classif_radius_fraction)).astype(int)
    if shuffle:
        np.random.seed(seed)
        shuffled = np.random.permutation(range(X.shape[0]))
        X = X[shuffled]
        y = y[shuffled].reshape(-1,1)
    return (X, y)