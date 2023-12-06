import inspect
import torch

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if inspect.isfunction(d) else d

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr