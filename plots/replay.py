import matplotlib.pyplot as plt
import matplotlib.animation as animation
from operator import itemgetter 
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import matplotlib.ticker as ticker

FeatureSpaceData = namedtuple('FeatureSpaceData', ['line', 'bent_line', 'prediction', 'target'])
FeatureSpaceLines = namedtuple('FeatureSpaceLines', ['grid', 'input', 'contour'])

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
        return self

    def _prepare_plot(self):
        # bent_contour_lines.shape: [1,1000,1000,2]
        if self.scale_fixed:
            xlim = [self.bent_contour_lines[:,:,:,0].min()-0.05, self.bent_contour_lines[:,:,:,0].max()+0.05]
            ylim = [self.bent_contour_lines[:,:,:,1].min()-0.05, self.bent_contour_lines[:,:,:,1].max()+0.05]
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        self.ax.set_xlabel(r"$x_0$", fontsize=12)
        self.ax.set_ylabel(r"$x_1$", fontsize=12, rotation=0)
        # grid_lines.shape: [22,1000,2]
        self.lines = []
        self.points = []
        for c in range(self.grid_lines.shape[0]):
            # the ',' is because the ax.plot return a [Line2D]
            line, = self.ax.plot([], [], linewidth=0.5, color='k')
            self.lines.append(line)
        # self.classes: [0]
        for c in range(len(self.classes)):
            point = self.ax.scatter([], [])
            self.points.append(point)
        
        # self.bent_contour_lines.shape: [1, 1000, 1000, 2]
        contour_x = self.bent_contour_lines[0, :, :, 0] 
        contour_y = self.bent_contour_lines[0, :, :, 1]
        
        if self.boundary: # False
            self.contour = self.ax.contourf(contour_x, contour_y, np.zeros(shape=(len(contour_x), len(contour_y))),
                cmap=plt.cm.brg, alpha=self.alpha, levels = np.linspace(0,1,8))
    
    # i=0, fs: this
    @staticmethod 
    def _update(i, fs, epoch_start=0, colors=None, **kwargs):
        epoch = i + epoch_start
        fs.ax.set_title('Epoch: {}'.format(epoch))
        if not fs.scale_fixed:
            xlim = [fs.bent_contour_lines[epoch,:,:,0].min()-0.05, fs.bent_contour_lines[epoch,:,:,0].max()+0.05]
            ylim = [fs.bent_contour_lines[epoch,:,:,1].min()-0.05, fs.bent_contour_lines[epoch,:,:,1].max()+0.05]
            fs.ax.set_xlim(xlim)
            fs.ax.set_ylim(ylim)
        # len: 22
        if len(fs.lines):
            # fs.bent_lines[epoch].shape: [22,1000,2]
            # line_coords: [2,1000,22]            
            line_coords = fs.bent_lines[epoch].transpose()
        for c, line in enumerate(fs.lines):
            line.set_data(*line_coords[:,:,c]) # draw gridlines here
        if colors is None: 
            colors = ['r', 'b']
        if 's' not in kwargs.keys():
            kwargs.update({'s': 10}) # size
        if 'marker' not in kwargs.keys():
            kwargs.update({'marker': 'o'})
        
        # fs.bent_inputs: [1,1,22000,2]
        # input_coords: list of 1 element of [2,22000,1]
        input_coords = [coord[epoch].transpose() for coord in fs.bent_inputs]
        # fs.points, list of 1 of PathCollection objects
        # fs.classes: [0]        
        for c in range(len(fs.points)):
            fs.points[c].remove()
            fs.points[c] = fs.ax.scatter(*input_coords[c], color=colors[(fs.classes[c])], **kwargs)
        if fs.boundary: # False
            for c in fs.contour.collections:
                c.remove()
            fs.contour = fs.ax.contourf(
                fs.bent_contour_lines[epoch, :, :, 0],
                fs.bent_contour_lines[epoch, :, :, 1],
                fs.predictions[epoch].squeeze(),
                cmap=fs.cmap, alpha=fs.alpha, levels=np.linspace(0,1,8)
                )
        fs.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        fs.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        fs.ax.locator_params(tight=True, nbins=7)
        fs.ax.yaxis.set_label_coords(-0.15, 0.5)
        return fs.lines 
        
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

    if epoch_end == -1:
        epoch_end = len(states) - 1 # states: [OrderedDict([(input.weight,tensor), (input.bias,tensor)])]
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
        # {'input': [1,22000,2]}
        X_values = get_values_for_epoch(model, states, epoch, grid_lines.reshape(-1,2))
        bent_inputs.append(X_values[layer_name]) # [1,22000,2]
        if input_dims == 2 and display_grid:
            # {'input': [1,22000,2]}
            grid_values = get_values_for_epoch(model, states, epoch, grid_lines.reshape(-1,2))
            output_shape = (grid_lines.shape[:2]) + (-1,) # [22,1000, -1]
            bent_lines.append(grid_values[layer_name].reshape(output_shape)) # [22,1000,2]
            # {'input': [1,1000000,2]}
            contour_values = get_values_for_epoch(model, states, epoch, contour_lines.reshape(-1,2))
            output_shape = (contour_lines.shape[:2]) + (-1,) # [1000,1000,-1]
            bent_contour_lines.append(contour_values[layer_name].reshape(output_shape)) # [1000,1000,2]
            # Make predictions for each point in contour surface; 'func' is Sequential()
            bent_preds.append(( func(contour_values[last_layer_name]).reshape(output_shape) > 0.5 ).astype(int))
    bent_inputs = np.array(bent_inputs)
    bent_lines = np.array(bent_lines)
    bent_contour_lines = np.array(bent_contour_lines)
    bent_preds = np.array(bent_preds)
    # Convert arrays to namedtuple
    line_data = FeatureSpaceLines(grid=grid_lines, input=X, contour=contour_lines)
    bent_line_data = FeatureSpaceLines(grid=bent_lines, input=bent_inputs, contour=bent_contour_lines)
    _feature_space_data = FeatureSpaceData(
        line=line_data, 
        bent_line=bent_line_data, 
        prediction=bent_preds, target=y)
    return _feature_space_data

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

# x: reshaped grid_lines [1,22000,2]
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

