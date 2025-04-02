import matplotlib.plt as plt
import matplotlib.animation as animation
from operator import itemgetter 

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
        self.bent_inputs = [self.bent_inputs[:,self.targets==target, :] for target in self.clasess]
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