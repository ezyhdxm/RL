import torch
import torch.nn as nn
from typing import Union

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using GPU {gpu_id}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'identity': nn.Identity(),
    'selu': nn.SELU(),
    'elu': nn.ELU(),
    'softplus': nn.Softplus(),
}

class MLP(nn.Module):
    def __init__(self, input_size:int, output_size:int, n_layers:int, hidden_size:int, activation:Activation = 'relu', output_activation:Activation = 'identity'):
        super().__init__()
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        in_size = input_size
        for _ in range(n_layers):
            self.layers.append(nn.Linear(in_size, hidden_size))
            self.layers.append(activation)
            in_size = hidden_size
        
        self.layers.append(nn.Linear(in_size, output_size))
        self.layers.append(output_activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x