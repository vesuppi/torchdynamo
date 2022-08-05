import torch

def unary_op(x):
    t0 = torch.exp(x)
    y = torch.cos(t0)
    return y