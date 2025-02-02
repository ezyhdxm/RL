import torch

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    print(use_gpu)
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