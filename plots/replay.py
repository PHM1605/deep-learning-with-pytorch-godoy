import matplotlib.pyplot as plt
import matplotlib.animation as animation
from operator import itemgetter 
import torch
import torch.nn as nn
import numpy as np

# Basic plot class, NOT to be initiated directly
class Basic(object):
    def __init__(self, ax):
        self._title = ''
        self._custom_title = ''
        self.n_epochs = 0 
        self.ax = ax
        self.ax.clear()
        self.fig = ax.get_figure()
    
    @property
    def title(self):
        title = self._title 
        if not isinstance(title, tuple):
            title = (self._title,)
        title = tuple([' '.join([self._custom_title, t]) for t in title])
        return title 

    @property 
    def axes(self):
        return (self.ax,)
    
    def load_data(self, **kwargs):
        self._prepare_plot()
        return self 
    
    def _prepare_plot(self):
        pass 
    
    @staticmethod 
    def _update(i, object, epoch_start=0):
        pass 

    def set_title(self, title):
        self._custom_title = title 

    def plot(self, epoch):
        self.__class__._update(epoch, self)
        self.fig.tight_layout()
        plt.savefig('test.png')
        return self.fig 
    
    # return animation function for the data (from epoch_start to epoch_end)
    def animate(self, epoch_start=0, epoch_end=-1):
        if epoch_end == -1:
            epoch_end = self.n_epochs 
        anim = animation.FuncAnimation(self.fig, self.__class__.update, fargs=(self, epoch_start), frames=(epoch_end-epoch_start), blit=True)
        return anim 
    

class FeatureSpace(Basic):
    def __init__(self, ax, scale_fixed=True, boundary=True, cmap=None, alpha=1.0):
        super(FeatureSpace, self).__init__(ax)
        self.ax.grid(False)
        self.scale_fixed = scale_fixed # if True, axis scales are fixed to the maximum from beginning
        self.boundary = boundary 
        self.contour = None 
        self.bent_inputs = None 
        self.bent_lines = None 
        self.bent_contour_lines = None 
        self.grid_lines = None 
        self.counter_lines = None 
        self.predictions = None 
        self.targets = None 
        if cmap is None:
            cmap = plt.cm.RdBu
        self.cmap = cmap 
        self.alpha = alpha 
        self.n_inputs = 0
        self.lines = []
        self.points = []

    def load_data(self, feature_space_data):
        self.predictions = feature_space_data.prediction 
        self.targets = feature_space_data.target 
        self.grid_lines, self.inputs, self.contour_lines = feature_space_data.line
        self.bent_lines, self.bent_inputs, self.bent_contour_lines = feature_space_data.bent_line
        self.n_epochs = self.bent_inputs.shape[0]
        self.n_inputs = self.bent_inputs.shape[-1]
        self.classes = np.unique(self.targets)
        # many lists, each list is one target
        self.bent_inputs = [self.bent_inputs[:,self.targets==target, :] for target in self.classes]
        self._prepare_plot()

    def _prepare_plot(self):
        pass

# Build a FeatureSpace object for plotting and animating
# model: identity model, with only one Layer of Linear(2,2) with name 'input'
# states: [model.state_dict()]
# X: hidden state [1,1,2]
# y: point number i-th
# layer_name: 'input'
def build_feature_space(model, states, X, y, layer_name=None, 
    contour_points=1000, xlim=(-1,1), ylim=(-1,1), display_grid=True, epoch_start=0, epoch_end=-1):
    # layers: [('', Sequential), ('input', Linear), ('activation', Sigmoid)]
    layers = list(model.named_modules())
    last_layer_name, last_layer_class = layers[-1]
    is_logit = not isinstance(last_layer_class, nn.Sigmoid)
    if is_logit:
        activation_idx = -2 
        func = lambda x: 1/(1+np.exp(-x))
    else:
        activation_idx = -3
        func = lambda x:x 
    # take item 0 of each element of layers
    names = np.array(list(map(itemgetter(0), layers)))
    matches = names == layer_name
    if np.any(matches):
        activation_idx = np.argmax(matches)
    else:
        raise AttributeError("No layer named {}".format(layer_name))
    if layer_name is None:
        layer_name = layers[activation_idx][0]
    
    try:
        final_dims = layers[activation_idx][1].out_features
    except:
        try:
            final_dims = layers[activation_idx+1][1].in_features 
        except:
            final_dims = layers[activation_idx-1][1].out_features 
    assert final_dims == 2, 'Only layers with 2-dimensinal outputs are supported!'
    
    y_ind = np.atleast_1d(y.squeeze().argsort()) # sort points in ascending order
    X = np.atleast_2d(X.squeeze())[y_ind].reshape(X.shape) # X: [1,1,2]
    y = np.atleast_1d(y.squeeze())[y_ind]

    print("STATES: ", len(states))
    if epoch_end == -1:
        epoch_end = len(states) - 1
    epoch_end = min(epoch_end, len(states)-1) # 1
    input_dims = X.shape[-1] # 2
    n_classes = len(np.unique(y)) # 1

    #Build grid & contour
    grid_lines = np.array([])
    contour_lines = np.array([])
    if input_dims == 2 and display_grid:
        grid_lines = build_2d_grid(xlim, ylim) #[22,1000,2]
        contour_lines = build_2d_grid(xlim, ylim, contour_points, contour_points) # [1000,1000,2]
    
    bent_lines = []
    bent_inputs = []
    bent_contour_lines = []
    bent_preds = []
    for epoch in range(epoch_start, epoch_end+1):
        X_values = get_values_for_epoch(model, states, epoch, grid_lines.reshape(-1,2))

    return

# create a 2D grid of 'n_lines' of 'n_points' each
def build_2d_grid(xlim, ylim, n_lines=11, n_points=1000):
    xs = np.linspace(*xlim, num=n_lines)
    ys = np.linspace(*ylim, num=n_points)
    x0, y0 = np.meshgrid(xs, ys)
    lines_x0 = np.atleast_3d(x0.transpose()) # [11,1000,1], rows [-1,-1...], [-0.8,-0.8...]
    lines_y0 = np.atleast_3d(y0.transpose()) # [11,1000,1], rows [-1,-0.998...], [-1,-0.998...]
    
    xs = np.linspace(*xlim, num=n_points) # [-1,-0.998..,0.998,1]
    ys = np.linspace(*ylim, num=n_lines) # [-1.-0.8...,0.8,1]
    x1, y1 = np.meshgrid(xs, ys)
    lines_x1 = np.atleast_3d(x1) # [11,1000,1], rows (the same) [-1,-0.99,...,1]
    lines_y1 = np.atleast_3d(y1) # [11,1000,1], rows [-1,...,-1], [-0.8,...,-0.8]
    
    vertical_lines = np.concatenate([lines_x0, lines_y0], axis=2) # [11,1000,2]
    horizontal_lines = np.concatenate([lines_x1, lines_y1], axis=2) # [11,1000,2]
    if n_lines != n_points:
        lines = np.concatenate([vertical_lines, horizontal_lines], axis=0) # [22,1000,2]
    else:
        lines = vertical_lines
    return lines 

def get_values_for_epoch(model, states, epoch, x):
    with torch.no_grad():
        model.load_state_dict(states[epoch])
    return get_intermediate_values(model, x)

def get_intermediate_values(model, x):
    hooks = {}
    visualization = {}
    layer_names = {} # {Layer: 'layer1'}

    def hook_fn(m, i, o):
        visualization[layer_names[m]] = o.cpu().detach().numpy()
    
    for name, layer in model.named_modules():
        if name != '':
            layer_names[layer] = name 
            hooks[name] = layer.register_forward_hook(hook_fn)
    device = list(model.parameters())[0].device.type
    model(torch.as_tensor(x).float().unsqueeze(0).to(device))

    for hook in hooks.values():
        hook.remove()
    
    return visualization

