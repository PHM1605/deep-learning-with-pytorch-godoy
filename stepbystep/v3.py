import numpy as np
import datetime, random, torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import Normalize
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR

plt.style.use('fivethirtyeight')

class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn 
        self.optimizer = optimizer 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.train_loader = None 
        self.val_loader = None 
        self.writer = None
        self.scheduler = None 
        self.is_batch_lr_scheduler = False 

        self.losses = [] 
        self.val_losses = []
        self.learning_rates = []
        self.total_epochs = 0
        
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()
        ## Hook
        self.visualization = {}
        self.handles = {} 
        ## Gradients tracking
        self._gradients = {}
    
    def to(self, device):
        try:
            self.device = device 
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couln't send it to {device}, sending it to {self.deivce} instead.")
            self.model.to(self.device)
    
    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader 
    
    def set_tensorboard(self, name, folder='runs'):
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')
    
    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            # Compute grad for parameters
            loss.backward()
            # Update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            return loss.item()
        return perform_val_step_fn
    
    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader 
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader 
            step_fn = self.train_step_fn 
        if data_loader is None:
            return None 
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        loss = np.mean(mini_batch_losses)
        return loss 
    
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass 
    
    def train(self, n_epochs, seed=42):
        self.set_seed(seed)
        for epoch in range(n_epochs):
            self.total_epochs += 1
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                self.writer.add_scalars(main_tag='loss', tag_scalar_dict=scalars, global_step=epoch)
        if self.writer:
            self.writer.close()

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'val_loss': self.val_losses 
        }        
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train() 
    
    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()
    
    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('test.png')
        return fig 
    
    def add_graph(self):
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))
    
    def count_parameters(self):
        return sum(p.numel()
                   for p in self.model.parameters() 
                   if p.requires_grad)
    
    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None, layer_name='', title=None):
        n_images = len(axs) # n_images = in_channels 
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            if title is not None:
                ax.set_title(f'{title} #{j}', fontsize=12)
            shp = np.atleast_2d(image).shape 
            ax.set_ylabel(f'{layer_name}\n{shp[0]}x{shp[1]}')
            xlabel1 = '' if y is None else f'\nLabel: {y[j]}'
            xlabel2 = '' if yhat is None else f'\nPredicted: {yhat[j]}'
            xlabel = f'{xlabel1}{xlabel2}'
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap = 'gray',
                vmin = minv,
                vmax = maxv
            )
    
    def visualize_filters(self, layer_name, **kwargs):
        try: 
            layer = self.model 
            for name in layer_name.split("."):
                layer = getattr(layer, name)
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data.cpu().numpy() # [out_channels, in_channels, kernel_size, kernel_size]
                n_filters, n_channels, _, _ = weights.shape 
                size = (2*n_channels+2, 2*n_filters)
                fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
                axes = np.atleast_2d(axes)
                axes = axes.reshape(n_filters, n_channels)
                for i in range(n_filters):
                    StepByStep._visualize_tensors(axes[i, :], weights[i], layer_name=f'Filter #{i}', title='Channel')
                for ax in axes.flat:
                    ax.label_outer()
                fig.tight_layout()
                plt.savefig('test.png')
                return fig 
        except AttributeError:
            return 
    
    def attach_hooks(self, layers_to_hook, hook_fn=None):
        # Clear any previous values
        self.visualization = {}
        modules = list(self.model.named_modules()) # [('', Sequential), ('conv1', Conv2d), ('relu1', ReLU), ()....]
        layer_names = {layer: name for name, layer in modules[1:]} # {Conv2d:'conv1', ReLU:'relu1'}
        if hook_fn is None:
            def hook_fn(layer, inputs, outputs):
                name = layer_names[layer]
                values = outputs.detach().cpu().numpy()
                # if predictions for many times -> concatenate them
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])
        for name, layer in modules:
            if name in layers_to_hook:
                self.visualization[name] = None
                self.handles[name] = layer.register_forward_hook(hook_fn) # {'conv1':RemovableHandle 0x764aa, 'relu1':RemovableHandle 0x24a123}
    
    def remove_hooks(self):
        for handle in self.handles.values():
            handle.remove()
        self.handles = {}
    
    # layers: list of names ['conv1', 'relu1', ...]
    def visualize_outputs(self, layers, n_images=10, y=None, yhat=None):
        # condition of filter: l, which is each component of "layers", is in the list of keys stored in "visualization"
        layers = filter(lambda l: l in self.visualization.keys(), layers)
        layers = list(layers)
        shapes = [self.visualization[layer].shape for layer in layers]
        # shape: [batch, n_channels, width, height] or [batch, length]
        # n_rows: #channels for each layer
        n_rows = [shape[1] if len(shape)==4 else 1 
                  for shape in shapes]
        total_rows = np.sum(n_rows)
        fig, axes = plt.subplots(total_rows, n_images, figsize=(1.5*n_images, 1.5*total_rows))
        axes = np.atleast_2d(axes).reshape(total_rows, n_images)
        row = 0
        for i, layer in enumerate(layers):
            start_row = row 
            # output: [batch, n_channels, width, height] or [batch, length]
            output = self.visualization[layer]
            is_vector = len(output.shape)==2
            # Loop through each layer
            for j in range(n_rows[i]):
                StepByStep._visualize_tensors(
                    axes[row, :],                    
                    output if is_vector else output[:,j].squeeze(),
                    y,
                    yhat,
                    layer_name=layers[i] if is_vector else f'{layers[i]}\nfil#{row-start_row}',
                    title='Image' if (row==0) else None
                )
                row += 1
        for ax in axes.flat:
            ax.label_outer()
        plt.tight_layout()
        plt.savefig('test.png')
        return fig 

    # Check for only one batch
    def correct(self, x, y, threshold=0.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()
        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # In binary classification, check if the last layer is sigmoid or logits 
            if isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid):
                predicted = (yhat > threshold).long()
            else:
                predicted = (F.sigmoid(yhat) > threshold).long()
        result = []
        for c in range(n_dims):
            n_class = (y==c).sum().item() # #samples with true class 'c'
            n_correct = (predicted[y==c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result) # [#classes, 2], row0: #correct of class0, #true of class0
    
    # Apply 'func' for the whole list of batches
    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(x,y) for i,(x,y) in enumerate(loader)]
        results = torch.stack(results, axis=0)
        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)
        return results 

    @staticmethod
    def statistics_per_channel(images, labels):
        n_samples, n_channels, n_height, n_weight = images.size()
        # Flatten height & width to a single dimension
        flatten_per_channel = images.reshape(n_samples, n_channels, -1) # [batch,3,784]
        means = flatten_per_channel.mean(axis=2) # [batch,3]
        stds = flatten_per_channel.std(axis=2) # [batch,3]
        sum_means = means.sum(axis=0) # [3]
        sum_stds = stds.sum(axis=0) # [3]
        n_samples = torch.tensor([n_samples]*n_channels).float() # [n_samples,n_samples,n_samples]
        return torch.stack([n_samples, sum_means, sum_stds]) # 1st row:#samples of each channel; 2nd row:mean of each channel; 3rd:std of each channel
    
    @staticmethod 
    def make_normalizer(loader):
        total_samples, total_means, total_stds = StepByStep.loader_apply(loader, StepByStep.statistics_per_channel)
        norm_mean = total_means / total_samples # [3], one for each channel
        norm_std = total_stds / total_samples # [3], one for each channel 
        return Normalize(mean=norm_mean, std=norm_std)
    
    def lr_range_test(self, data_loader, end_lr, num_iter=100, step_mode='exp', alpha=0.05, ax=None):
        previous_states = {
            'model': deepcopy(self.model.state_dict()),
            'optimizer': deepcopy(self.optimizer.state_dict())
        }
        # Retrieve the learning rate set in the optimizer
        start_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        lr_fn = make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)
        tracking = {'loss': [], 'lr': []}
        iteration = 0

        # If we need more iterations than #batches in the DataLoader, DataLoader is looped more than once
        while (iteration < num_iter):
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                yhat = self.model(x_batch)
                loss = self.loss_fn(yhat, y_batch)
                loss.backward()
                tracking['lr'].append(scheduler.get_last_lr()[0])
                # Smooting the curve of losses while tracking
                if iteration == 0:
                    tracking['loss'].append(loss.item())
                else:
                    prev_loss = tracking['loss'][-1]
                    smoothed_loss = alpha * loss.item() + (1-alpha) * prev_loss
                    tracking['loss'].append(smoothed_loss)
                iteration += 1
                if iteration == num_iter:
                    break
                self.optimizer.step()
                scheduler.step() # change the learning rate to the next new one
                self.optimizer.zero_grad()
        
        # Restores the original state
        self.optimizer.load_state_dict(previous_states['optimizer'])
        self.model.load_state_dict(previous_states['model'])

        # Plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6,4))
        else:
            fig = ax.get_figure()
        ax.plot(tracking['lr'], tracking['loss'])
        if step_mode == 'exp':
            ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        plt.savefig('test.png')
        return tracking, fig
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer 

    # layers_to_hook: ['conv1','maxpool1',...] or 'conv1'
    def capture_gradients(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]
        modules = list(self.model.named_modules())
        self._gradients = {} 

        def make_log_fn(name, parm_id):
            # receive gradients returns modified grads or 'None'
            def log_fn(grad):
                self._gradients[name][parm_id].append(grad.tolist())
                return None
            return log_fn

        for name, layer in self.model.named_modules():
            if name in layers_to_hook:
                self._gradients.update({name: {}}) # {'conv1': {}}
                for parm_id, p in layer.named_parameters(): # {'conv1': {'weight': [], 'bias'}}
                    if p.requires_grad:
                        self._gradients[name].update({parm_id:[]})
                        log_fn = make_log_fn(name, parm_id)
                        self.handles[f'{name}.{parm_id}.grad'] = p.register_hook(log_fn)
        return 

    def capture_parameters(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]
        modules = list(self.model.named_modules()) # {'conv1':Conv2D, 'maxpool':MaxPool2D}
        layer_names = {layer: name for name, layer in modules} # e.g. {Conv2D:'conv1', MaxPool2D:'maxpool'}
        self._parameters = {}
        for name, layer in modules:
            if name in layers_to_hook:
                self._parameters.update({name: {}})
                for parm_id, p in layer.named_parameters():
                    self._parameters[name].update({parm_id: []})

        def fw_hook_fn(layer, inputs, outputs):
            name = layer_names[layer] # 'conv2d'
            # parm_id: 'bias', parameter: [0.34]
            for parm_id, parameter in layer.named_parameters():
                self._parameters[name][parm_id].append(
                    parameter.tolist()
                )
                
        self.attach_hooks(layers_to_hook, fw_hook_fn)
        return

def make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp'):
    # iteration (list): [0,1,2,...10]
    if step_mode == 'linear':
        factor = (end_lr/start_lr - 1) / num_iter 
        def lr_fn(iteration):
            return 1 + iteration * factor
    else:
        factor = (np.log(end_lr) - np.log(start_lr)) / num_iter
        def lr_fn(iteration):
            return np.exp(factor)**iteration 
    return lr_fn 